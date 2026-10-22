"""Count captions that exceed the text context length used for embeddings."""

import argparse
from pathlib import Path

import open_clip
import pandas as pd


DEFAULT_METADATA = Path("embeddings/random_samples/metadata.csv")
MODEL_NAME = "ViT-B-32"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help=f"CSV to inspect (default: {DEFAULT_METADATA})")
    parser.add_argument("--output", type=Path,
                        help="Optional CSV path for captions exceeding the limit")
    args = parser.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata}")

    metadata = pd.read_csv(args.metadata)
    caption_column = next(
        (name for name in ("caption", "Caption", "text") if name in metadata.columns),
        None,
    )
    if caption_column is None:
        raise ValueError(f"No caption column found in {args.metadata}. Columns: {list(metadata.columns)}")

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    context_length = getattr(tokenizer, "context_length", 77) or 77

    # SimpleTokenizer.encode returns content token IDs without the start/end
    # markers. Those markers consume two positions in the model context.
    content_limit = context_length - 2
    token_counts = []
    for caption in metadata[caption_column].fillna("").astype(str):
        token_counts.append(len(tokenizer.encode(caption)))

    report = metadata.copy()
    report["content_token_count"] = token_counts
    report["exceeds_context_limit"] = report["content_token_count"] > content_limit
    over_limit = report[report["exceeds_context_limit"]].copy()

    total = len(report)
    count = len(over_limit)
    percent = (100 * count / total) if total else 0.0
    print(f"Metadata file: {args.metadata}")
    print(f"CLIP context length: {context_length} tokens including start/end markers")
    print(f"Maximum caption content length: {content_limit} tokens")
    print(f"Captions checked: {total}")
    print(f"Captions over limit: {count} ({percent:.1f}%)")
    if token_counts:
        print(f"Longest caption: {max(token_counts)} content tokens")

    if count:
        id_column = next((name for name in ("image_id", "image", "id") if name in report.columns), None)
        print("\nOver-limit captions:")
        for _, row in over_limit.iterrows():
            identifier = row[id_column] if id_column else row.name
            print(f"  {identifier}: {int(row['content_token_count'])} tokens")
    else:
        print("\nNo captions exceed the limit.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        over_limit.to_csv(args.output, index=False)
        print(f"\nSaved over-limit captions to {args.output}")


if __name__ == "__main__":
    main()
