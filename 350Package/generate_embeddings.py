# py generate_embeddings.py
import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

CSV_PATH = Path("data/random_samples.csv")

IMAGE_DIR = Path("data/images")

OUTPUT_DIR = Path("embeddings/random_samples")

BATCH_SIZE = 32

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


# ============================================================
# 2. SELECT DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. LOAD CLIP
# ============================================================

print("\n" + "=" * 55)
print("       POMPEII CLIP EMBEDDING GENERATOR")
print("=" * 55)

print(f"\nDevice: {device}")
print(f"Model:  {MODEL_NAME}")

print("\nLoading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(device)
model.eval()

print("CLIP loaded successfully.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_PATH.resolve()}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image_id",
    "image_path",
    "caption",
    "query_file"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"CSV is missing required columns: {missing_columns}"
    )

total = len(df)

if total == 0:
    raise ValueError("CSV contains no records.")

print(f"\nLoaded {total} image-caption pairs.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. FIND IMAGE FILE
# ============================================================

def find_image(image_path):
    """
    Locate an image using its CSV image_path.

    Supports paths with or without extensions.
    """

    image_path = Path(str(image_path))

    # Check exact path
    if image_path.exists():
        return image_path

    # Check relative to working directory
    relative_path = Path.cwd() / image_path

    if relative_path.exists():
        return relative_path

    # Check image directory using filename
    filename = image_path.name

    candidates = list(
        IMAGE_DIR.glob(filename + ".*")
    )

    if candidates:
        return candidates[0]

    # Check filename without extension
    if image_path.suffix:

        candidates = list(
            IMAGE_DIR.glob(image_path.stem + ".*")
        )

        if candidates:
            return candidates[0]

    return None


# ============================================================
# 6. GENERATE EMBEDDINGS
# ============================================================

all_image_embeddings = []
all_text_embeddings = []

successful_metadata = []

failed = 0
processed = 0

num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

print("\nStarting embedding generation...\n")

for batch_number, start in enumerate(
    range(0, total, BATCH_SIZE),
    start=1
):

    batch = df.iloc[start:start + BATCH_SIZE]

    images = []
    captions = []
    valid_rows = []

    # --------------------------------------------------------
    # Load images and captions
    # --------------------------------------------------------

    for index, row in batch.iterrows():

        image_path = find_image(row["image_path"])

        if image_path is None:
            failed += 1
            continue

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")
                processed_image = preprocess(image)

            images.append(processed_image)

            captions.append(str(row["caption"]))

            valid_rows.append(index)

        except Exception:
            failed += 1
            continue

    # --------------------------------------------------------
    # Encode valid batch
    # --------------------------------------------------------

    if images:

        image_input = torch.stack(images).to(device)

        text_input = tokenizer(captions).to(device)

        with torch.no_grad():

            image_embeddings = model.encode_image(image_input)

            text_embeddings = model.encode_text(text_input)

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1,
            keepdim=True
        )

        text_embeddings = text_embeddings / text_embeddings.norm(
            dim=-1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Store embeddings
        # ----------------------------------------------------

        all_image_embeddings.append(
            image_embeddings.cpu().numpy()
        )

        all_text_embeddings.append(
            text_embeddings.cpu().numpy()
        )

        successful_metadata.append(
            df.loc[valid_rows].copy()
        )

    # --------------------------------------------------------
    # Update progress
    # --------------------------------------------------------

    processed = min(start + len(batch), total)

    successful = sum(
        len(x) for x in successful_metadata
    )

    percentage = processed / total * 100

    print(
        f"\rProgress: {processed:>3}/{total} "
        f"({percentage:5.1f}%) | "
        f"Successful: {successful} | "
        f"Failed: {failed} | "
        f"Batch: {batch_number}/{num_batches}",
        end="",
        flush=True
    )


# ============================================================
# 7. CHECK RESULTS
# ============================================================

if not all_image_embeddings:

    print("\n")

    raise RuntimeError(
        "No images were successfully processed. "
        "Check data/images/ and the CSV image paths."
    )


# ============================================================
# 8. COMBINE BATCHES
# ============================================================

image_embeddings = np.vstack(all_image_embeddings)

text_embeddings = np.vstack(all_text_embeddings)

metadata = pd.concat(
    successful_metadata,
    ignore_index=True
)


# ============================================================
# 9. SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_DIR / "image_embeddings.npy",
    image_embeddings
)

np.save(
    OUTPUT_DIR / "text_embeddings.npy",
    text_embeddings
)


# ============================================================
# 10. SAVE METADATA
# ============================================================

metadata.to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 55)
print("          EMBEDDING GENERATION COMPLETE")
print("=" * 55)

print(f"\nTotal pairs:       {total}")
print(f"Successful:        {len(metadata)}")
print(f"Failed:            {failed}")

print("\n" + "-" * 55)

print(f"Image embeddings:  {image_embeddings.shape}")
print(f"Text embeddings:   {text_embeddings.shape}")

print("\n" + "-" * 55)

print(f"Output directory:  {OUTPUT_DIR}")

print("\nSaved files:")

print("  - image_embeddings.npy")
print("  - text_embeddings.npy")
print("  - metadata.csv")

print("\n" + "=" * 55)
print("                  DONE")
print("=" * 55)