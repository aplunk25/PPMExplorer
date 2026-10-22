# py download_images.py

import csv
import json
import time
import mimetypes
import requests

from pathlib import Path
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = Path("data/random_samples.csv")
JSON_DIR = Path("data/queries")
OUTPUT_DIR = Path("data/images")

DELAY_SECONDS = 1.0
TIMEOUT = 30

URL_FIELDS = {
    "url",
    "image_url",
    "imageurl",
    "source_url",
    "sourceurl",
    "download_url",
    "downloadurl",
    "original_url",
    "originalurl",
    "href",
    "src",
    "image",
}

ID_FIELDS = {
    "image_id",
    "imageid",
    "id",
    "key",
    "image_path",
    "imagepath",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(value):
    return str(value).strip().lower()


def load_json(json_path):

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"[ERROR] Could not read JSON: {e}")
        return None


def find_image_url(data, image_id):
    """
    Recursively searches JSON data for the specified image ID
    and retrieves its associated URL.
    """

    target_id = normalize(image_id)

    def search(obj):

        if isinstance(obj, dict):

            # Check whether this dictionary represents our image
            matches = any(
                normalize(obj.get(key, "")) == target_id
                for key in ID_FIELDS
            )

            if matches:

                # Search this matching record for a URL
                def extract_url(record):

                    if isinstance(record, dict):

                        for key, value in record.items():

                            if normalize(key) in URL_FIELDS:

                                if isinstance(value, str):
                                    if value.startswith(("http://", "https://")):
                                        return value

                        for value in record.values():

                            result = extract_url(value)

                            if result:
                                return result

                    elif isinstance(record, list):

                        for item in record:

                            result = extract_url(item)

                            if result:
                                return result

                    return None

                result = extract_url(obj)

                if result:
                    return result

            # Check whether the image ID is itself a dictionary key
            if target_id in [normalize(k) for k in obj.keys()]:

                for key, value in obj.items():

                    if normalize(key) == target_id:
                        result = search(value)

                        if result:
                            return result

            # Search nested dictionaries and lists
            for value in obj.values():

                result = search(value)

                if result:
                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = search(item)

                if result:
                    return result

        return None

    return search(data)


def get_extension(url, content_type):

    path = urlparse(url).path
    extension = Path(path).suffix.lower()

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
        ".bmp",
        ".gif",
    }

    if extension in valid_extensions:
        return extension

    if content_type:

        extension = mimetypes.guess_extension(
            content_type.split(";")[0].strip()
        )

        if extension:
            return extension

    return ".jpg"


def download_image(session, image_url, image_id):

    response = session.get(
        image_url,
        timeout=TIMEOUT,
        stream=True
    )

    response.raise_for_status()

    extension = get_extension(
        image_url,
        response.headers.get("Content-Type", "")
    )

    output_path = OUTPUT_DIR / f"{image_id}{extension}"

    try:

        with open(output_path, "wb") as image_file:

            for chunk in response.iter_content(chunk_size=8192):

                if chunk:
                    image_file.write(chunk)

        if output_path.stat().st_size == 0:

            output_path.unlink()
            raise ValueError("Downloaded image is empty.")

        return output_path

    finally:
        response.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Verify directories and files
    # --------------------------------------------------------

    if not CSV_FILE.exists():

        print(f"[ERROR] CSV file not found: {CSV_FILE.resolve()}")
        return

    if not JSON_DIR.exists():

        print(f"[ERROR] JSON directory not found: {JSON_DIR.resolve()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        samples = list(csv.DictReader(f))

    if not samples:

        print("[ERROR] CSV contains no samples.")
        return

    # Verify expected columns
    required_columns = {"image_id", "query_file"}

    available_columns = set(samples[0].keys())

    if not required_columns.issubset(available_columns):

        print("[ERROR] CSV is missing required columns.")
        print(f"Required: {required_columns}")
        print(f"Found: {available_columns}")
        return

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("=" * 65)
    print("POMPEII IMAGE DOWNLOADER")
    print("=" * 65)

    print(f"CSV:              {CSV_FILE}")
    print(f"JSON directory:   {JSON_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total samples:    {len(samples)}")
    print(f"Download delay:   {DELAY_SECONDS} seconds")

    print("=" * 65)

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    json_cache = {}

    successful = 0
    skipped = 0
    failed = 0

    session = requests.Session()

    session.headers.update({
        "User-Agent": "PompeiiImageDownloader/1.0"
    })

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    try:

        for index, row in enumerate(samples, start=1):

            image_id = row["image_id"].strip()
            json_filename = row["query_file"].strip()

            print(f"\n[{index}/{len(samples)}] {image_id}")

            try:

                # --------------------------------------------
                # Locate JSON file
                # --------------------------------------------

                json_path = Path(json_filename)

                if not json_path.is_absolute():

                    if json_path.exists():
                        pass

                    else:
                        json_path = JSON_DIR / json_path.name

                if not json_path.exists():

                    print(f"[FAILED] JSON not found: {json_filename}")
                    failed += 1
                    continue

                # --------------------------------------------
                # Load JSON (cached)
                # --------------------------------------------

                json_key = str(json_path.resolve())

                if json_key not in json_cache:

                    print(f"[JSON] Loading {json_path.name}")

                    json_cache[json_key] = load_json(json_path)

                json_data = json_cache[json_key]

                if json_data is None:

                    failed += 1
                    continue

                # --------------------------------------------
                # Find image URL
                # --------------------------------------------

                image_url = find_image_url(
                    json_data,
                    image_id
                )

                if not image_url:

                    print("[FAILED] Could not find matching image URL.")
                    failed += 1
                    continue

                print(f"[URL] {image_url}")

                # --------------------------------------------
                # Check existing downloads
                # --------------------------------------------

                existing_files = list(
                    OUTPUT_DIR.glob(f"{image_id}.*")
                )

                if any(
                    file.stat().st_size > 0
                    for file in existing_files
                ):

                    print("[SKIPPED] Image already downloaded.")
                    skipped += 1
                    continue

                # --------------------------------------------
                # Download
                # --------------------------------------------

                output_path = download_image(
                    session,
                    image_url,
                    image_id
                )

                successful += 1

                size_kb = output_path.stat().st_size / 1024

                print(f"[SUCCESS] Saved: {output_path}")
                print(f"[SIZE] {size_kb:.2f} KB")

            except Exception as e:

                print(f"[FAILED] {e}")
                failed += 1

            finally:

                # --------------------------------------------
                # Rate limiting
                # --------------------------------------------

                if index < len(samples):

                    time.sleep(DELAY_SECONDS)

    finally:

        session.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DOWNLOAD COMPLETE")
    print("=" * 65)

    print(f"Total samples:       {len(samples)}")
    print(f"Successfully saved:  {successful}")
    print(f"Already existed:     {skipped}")
    print(f"Failed:              {failed}")

    print(f"\nImages saved to: {OUTPUT_DIR.resolve()}")

    print("=" * 65)


if __name__ == "__main__":
    main()