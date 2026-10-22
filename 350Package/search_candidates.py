"""Radius search over paired image and caption nodes in the shared CLIP space.

Every image and caption whose cosine distance from the query is within the
chosen radius is included. Its paired image/caption node is included too,
even when that linked node falls outside the radius.
"""

import argparse
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch


DEFAULT_EMBEDDINGS_DIR = Path("embeddings/random_samples")
DEFAULT_CAPTION_RADIUS = 0.30
DEFAULT_IMAGE_MIN_SIMILARITY = 0.22
CAPTION_DISPLAY_LIMIT = 240


def find_column(frame, candidates, required=True):
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(
            f"Could not find any of {candidates} in metadata columns "
            f"{list(frame.columns)}"
        )
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Text query; omit for interactive mode")
    parser.add_argument("--image-radius", type=float,
                        default=1.0 - DEFAULT_IMAGE_MIN_SIMILARITY,
                        help=f"Image cosine-distance radius, 0 to 2 "
                             f"(default: {1.0 - DEFAULT_IMAGE_MIN_SIMILARITY:.2f}; "
                             f"similarity >= {DEFAULT_IMAGE_MIN_SIMILARITY:.2f})")
    parser.add_argument("--caption-radius", type=float, default=DEFAULT_CAPTION_RADIUS,
                        help=f"Caption cosine-distance radius, 0 to 2 "
                             f"(default: {DEFAULT_CAPTION_RADIUS}; similarity >= 0.70)")
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output", type=Path,
                        help="Optional CSV output path (interactive mode appends query results)")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable results for a single query")
    args = parser.parse_args()

    if not 0 <= args.image_radius <= 2:
        parser.error("--image-radius must be between 0 and 2")
    if not 0 <= args.caption_radius <= 2:
        parser.error("--caption-radius must be between 0 and 2")

    directory = args.embeddings_dir
    image_file = directory / "image_embeddings.npy"
    text_file = directory / "text_embeddings.npy"
    metadata_file = directory / "metadata.csv"
    for path in (image_file, text_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}. Run generate_embeddings.py first.")

    image_vectors = np.load(image_file)
    caption_vectors = np.load(text_file)
    metadata = pd.read_csv(metadata_file)
    if len(metadata) != len(image_vectors) or len(metadata) != len(caption_vectors):
        raise ValueError("Metadata and embedding row counts do not match.")
    if len(metadata) == 0:
        raise ValueError("No embeddings were found.")
    if image_vectors.ndim != 2 or caption_vectors.ndim != 2:
        raise ValueError("Embedding files must contain 2D arrays.")
    if image_vectors.shape != caption_vectors.shape:
        raise ValueError("Image and caption embeddings must have matching dimensions.")

    id_column = find_column(metadata, ["image_id", "image", "id"], required=False)
    path_column = find_column(metadata, ["image_path", "path"], required=False)
    caption_column = find_column(metadata, ["caption", "text"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP ViT-B-32 (OpenAI weights) on {device}...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()

    image_vectors = torch.as_tensor(image_vectors, dtype=torch.float32, device=device)
    caption_vectors = torch.as_tensor(caption_vectors, dtype=torch.float32, device=device)
    image_vectors = torch.nn.functional.normalize(image_vectors, dim=-1)
    caption_vectors = torch.nn.functional.normalize(caption_vectors, dim=-1)

    def retrieve(query):
        tokens = tokenizer([query]).to(device)
        with torch.no_grad():
            query_vector = model.encode_text(tokens)
            query_vector = torch.nn.functional.normalize(query_vector, dim=-1)
            image_scores = (image_vectors @ query_vector.T).squeeze(1).cpu().numpy()
            caption_scores = (caption_vectors @ query_vector.T).squeeze(1).cpu().numpy()

        # For normalized vectors, cosine distance is 1 - cosine similarity.
        image_min_similarity = 1.0 - args.image_radius
        caption_min_similarity = 1.0 - args.caption_radius
        image_inside = image_scores >= image_min_similarity
        caption_inside = caption_scores >= caption_min_similarity
        included = image_inside | caption_inside

        # The image and caption in each metadata row are linked. Include both
        # whenever either one lies in the query radius.
        indices = np.flatnonzero(included)
        indices = sorted(
            indices,
            key=lambda i: max(float(image_scores[i]), float(caption_scores[i])),
            reverse=True,
        )
        records = []
        for rank, index in enumerate(indices, start=1):
            row = metadata.iloc[index]
            image_is_inside = bool(image_inside[index])
            caption_is_inside = bool(caption_inside[index])
            record = {
                "rank": rank,
                "image_id": row[id_column] if id_column else index,
                "image_path": row[path_column] if path_column else "",
                "associated_caption": row[caption_column],
                "image_similarity": float(image_scores[index]),
                "caption_similarity": float(caption_scores[index]),
                "image_in_radius": image_is_inside,
                "caption_in_radius": caption_is_inside,
                "included_by": (
                    "image and caption" if image_is_inside and caption_is_inside
                    else "image" if image_is_inside else "caption"
                ),
                "linked_node_added_outside_radius": image_is_inside != caption_is_inside,
                "image_cosine_distance_radius": args.image_radius,
                "caption_cosine_distance_radius": args.caption_radius,
                "image_minimum_similarity": image_min_similarity,
                "caption_minimum_similarity": caption_min_similarity,
                "query": query,
            }
            records.append(record)
        return pd.DataFrame(records)

    def show_results(query, results):
        print(f"\n{len(results)} linked image-caption pairs within/linked to thresholds")
        print(f"   Image: cosine distance <= {args.image_radius:.3f} "
              f"(similarity >= {1.0 - args.image_radius:.3f})")
        print(f"   Caption: cosine distance <= {args.caption_radius:.3f} "
              f"(similarity >= {1.0 - args.caption_radius:.3f})")
        print(f"Query: {query}")
        if results.empty:
            print("No image or caption nodes fell within their thresholds.")
            print("\nSummary:")
            print("   Total linked image-caption pairs returned: 0")
            print("   Pulled by image node: 0")
            print("   Pulled by caption node: 0")
            print("   Top 5 results: none")
            return
        for _, row in results.iterrows():
            print(f"\n{int(row['rank'])}. {row['image_id']}  "
                  f"(image {row['image_similarity']:.4f}; "
                  f"caption {row['caption_similarity']:.4f})")
            if row["image_path"]:
                print(f"   Path: {row['image_path']}")
            caption = str(row["associated_caption"])
            if len(caption) > CAPTION_DISPLAY_LIMIT:
                caption = caption[:CAPTION_DISPLAY_LIMIT].rstrip() + "..."
            print(f"   Caption: {caption}")
            print(f"   In radius: image={row['image_in_radius']}, "
                  f"caption={row['caption_in_radius']}")
            if row["linked_node_added_outside_radius"]:
                print("   The linked image or caption was added from outside the radius.")

        # Count each returned pair once, by the in-radius node type(s) that
        # caused it to be selected. Pairs can be in both groups.
        image_pulled = int(results["image_in_radius"].sum())
        caption_pulled = int(results["caption_in_radius"].sum())
        both_pulled = int((results["image_in_radius"] & results["caption_in_radius"]).sum())
        image_only = image_pulled - both_pulled
        caption_only = caption_pulled - both_pulled
        print("\nSummary:")
        print(f"   Total linked image-caption pairs returned: {len(results)}")
        print(f"   Pulled by image node: {image_pulled} "
              f"({image_only} image-only, {both_pulled} also in caption radius)")
        print(f"   Pulled by caption node: {caption_pulled} "
              f"({caption_only} caption-only, {both_pulled} also in image radius)")
        print("   Top 5 results:")
        for _, row in results.head(5).iterrows():
            print(f"      {int(row['rank'])}. {row['image_id']} "
                  f"(image {row['image_similarity']:.4f}, "
                  f"caption {row['caption_similarity']:.4f}; "
                  f"inside via {row['included_by']})")

    queries = [args.query] if args.query else None
    if queries:
        results = retrieve(queries[0])
        if not args.json:
            show_results(queries[0], results)
        else:
            print("JSON_RESULT=" + results.to_json(orient="records"))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.output, index=False)
            print(f"\nSaved candidates to {args.output}")
        return

    print(f"\nRadius search ready (image radius {args.image_radius:.3f}; "
          f"caption radius {args.caption_radius:.3f}). Type 'quit' to exit.")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        results = retrieve(query)
        show_results(query, results)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
            print(f"Saved query candidates to {args.output}")


if __name__ == "__main__":
    main()
