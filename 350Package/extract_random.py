# py extract_random.py data/queries/ppmx_ai-vision-vectors_query_result-*.json --output data/random_samples.csv
import json
import csv
import random
import argparse
import glob
from pathlib import Path


def extract_entries(json_file, sample_size=10):
    """
    Extract image_id, image_path, caption, and query_file
    from a JSON file.

    Randomly selects up to sample_size complete entries.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []

    def find_entries(obj):
        """Recursively search JSON for relevant entries."""

        if isinstance(obj, dict):

            # Check for required fields
            if "id" in obj and "caption_en" in obj:

                image_id = obj["id"]
                caption = obj["caption_en"]

                if image_id and caption:

                    entries.append({
                        "image_id": image_id,
                        "image_path": f"data/images/{image_id}",
                        "caption": caption,
                        "query_file": Path(json_file).name
                    })

            else:
                # Search nested dictionaries
                for value in obj.values():
                    find_entries(value)

        elif isinstance(obj, list):
            for item in obj:
                find_entries(item)

    find_entries(data)

    # Randomly select up to sample_size entries
    selected = random.sample(
        entries,
        min(sample_size, len(entries))
    )

    print(
        f"{Path(json_file).name}: "
        f"Found {len(entries)} entries, "
        f"selected {len(selected)}."
    )

    return selected


def main():

    parser = argparse.ArgumentParser(
        description="Extract random image-caption pairs from JSON files."
    )

    parser.add_argument(
        "json_files",
        nargs="+",
        help="JSON files or wildcard patterns."
    )

    parser.add_argument(
        "--output",
        default="random_samples.csv",
        help="Output CSV filename."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random entries per JSON file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling."
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    # Expand wildcard patterns
    json_files = []

    for pattern in args.json_files:

        matches = glob.glob(pattern)

        if matches:
            json_files.extend(matches)
        else:
            print(f"WARNING: No files found matching: {pattern}")

    # Remove duplicate filenames
    json_files = sorted(set(json_files))

    if not json_files:
        print("ERROR: No JSON files found.")
        return

    print(f"\nProcessing {len(json_files)} JSON files...\n")

    all_entries = []

    # Process each JSON independently
    for json_file in json_files:

        entries = extract_entries(
            json_file,
            args.samples
        )

        all_entries.extend(entries)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export combined CSV
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "image_path",
                "caption",
                "query_file"
            ]
        )

        writer.writeheader()
        writer.writerows(all_entries)

    print("\n------------------------------")
    print("Extraction complete!")
    print(f"JSON files processed: {len(json_files)}")
    print(f"Total entries selected: {len(all_entries)}")
    print(f"Output file: {output_path}")
    print("------------------------------")


if __name__ == "__main__":
    main()