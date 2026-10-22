# py verify_embeddings.py
import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

EMBEDDINGS_DIR = os.path.join("embeddings", "random_samples")

IMAGE_EMBEDDINGS_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "text_embeddings.npy"
)

METADATA_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "metadata.csv"
)

REPORT_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "similarity_report.csv"
)

COSINE_THRESHOLD = 0.25


# ============================================================
# Helper functions
# ============================================================

def normalize_embeddings(embeddings):
    """Normalize vectors so their dot product equals cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def get_column(df, possible_names):
    """Find a metadata column using several possible names."""
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def print_summary(
    total_pairs,
    cosine_correct,
    cosine_incorrect,
    paired_correct,
    paired_incorrect,
    average_paired_similarity,
    average_highest_similarity
):
    """Display the summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal pairs: {total_pairs}")
    print(f"Cosine threshold: {COSINE_THRESHOLD:.4f}")

    print("\nBased on cosine similarity:")
    print(f"  Correct:   {cosine_correct}")
    print(f"  Incorrect: {cosine_incorrect}")

    print("\nBased on paired caption ranking:")
    print(f"  Correct:   {paired_correct}")
    print(f"  Incorrect: {paired_incorrect}")

    print(
        f"\nAverage paired similarity: "
        f"{average_paired_similarity:.4f}"
    )

    print(
        f"Average highest similarity: "
        f"{average_highest_similarity:.4f}"
    )

    print("=" * 80)


def print_result(row):
    """Print one image's similarity results."""

    print("\n" + "=" * 80)
    print(f"Image ID: {row['image_id']}")

    print(
        f"Original caption similarity: "
        f"{row['original_caption_similarity']:.4f}"
    )

    print(
        f"Highest caption similarity:  "
        f"{row['highest_caption_similarity']:.4f}"
    )

    print(f"Image path: {row['image_path']}")

    print("\nCaption by cosine assignment:")
    print(row["highest_scoring_caption"])

    print("\nReal caption by paired assignment:")
    print(row["original_caption"])

    print("=" * 80)


# ============================================================
# Main program
# ============================================================

def main():

    print("\nLoading embeddings and metadata...")

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        IMAGE_EMBEDDINGS_FILE,
        TEXT_EMBEDDINGS_FILE,
        METADATA_FILE
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print("\nERROR: Required file not found:")
            print(file_path)
            print("\nRun generate_embeddings.py first.")
            return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
    text_embeddings = np.load(TEXT_EMBEDDINGS_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    if len(image_embeddings) != len(text_embeddings):
        print("\nERROR: Image and text embedding counts do not match.")
        return

    if len(metadata) != len(image_embeddings):
        print("\nERROR: Metadata and embedding counts do not match.")
        return

    image_id_column = get_column(
        metadata,
        ["image_id", "Image ID", "id"]
    )

    image_path_column = get_column(
        metadata,
        ["image_path", "Image Path", "path"]
    )

    caption_column = get_column(
        metadata,
        ["caption", "Caption", "text"]
    )

    if not all([image_id_column, image_path_column, caption_column]):
        print("\nERROR: Could not find the required metadata columns.")
        print("Available columns:", list(metadata.columns))
        return

    # --------------------------------------------------------
    # Normalize embeddings
    # --------------------------------------------------------

    image_embeddings = normalize_embeddings(image_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)

    # --------------------------------------------------------
    # Calculate cosine similarity matrix
    # --------------------------------------------------------

    print("Calculating cosine similarity matrix...")

    similarity_matrix = np.matmul(
        image_embeddings,
        text_embeddings.T
    )

    # --------------------------------------------------------
    # Analyze each image
    # --------------------------------------------------------

    results = []

    for image_index in range(len(image_embeddings)):

        similarities = similarity_matrix[image_index]

        # Similarity to the original paired caption.
        original_caption_similarity = float(
            similarities[image_index]
        )

        # Highest-scoring caption for this image.
        highest_caption_index = int(np.argmax(similarities))

        highest_caption_similarity = float(
            similarities[highest_caption_index]
        )

        original_caption = metadata.iloc[
            image_index
        ][caption_column]

        highest_scoring_caption = metadata.iloc[
            highest_caption_index
        ][caption_column]

        image_id = metadata.iloc[
            image_index
        ][image_id_column]

        image_path = metadata.iloc[
            image_index
        ][image_path_column]

        # Does the original paired caption meet the threshold?
        cosine_status = (
            original_caption_similarity >= COSINE_THRESHOLD
        )

        # Is the original paired caption the highest-scoring caption?
        paired_status = (
            highest_caption_index == image_index
        )

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "original_caption": original_caption,
            "highest_scoring_caption": highest_scoring_caption,
            "original_caption_similarity": original_caption_similarity,
            "highest_caption_similarity": highest_caption_similarity,
            "highest_caption_index": highest_caption_index,
            "cosine_status": cosine_status,
            "paired_status": paired_status
        })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Calculate summary statistics
    # --------------------------------------------------------

    total_pairs = len(results_df)

    cosine_correct = int(results_df["cosine_status"].sum())
    cosine_incorrect = total_pairs - cosine_correct

    paired_correct = int(results_df["paired_status"].sum())
    paired_incorrect = total_pairs - paired_correct

    average_paired_similarity = results_df[
        "original_caption_similarity"
    ].mean()

    average_highest_similarity = results_df[
        "highest_caption_similarity"
    ].mean()

    # --------------------------------------------------------
    # Save full report
    # --------------------------------------------------------

    results_df.to_csv(REPORT_FILE, index=False)

    print(f"\nFull similarity report saved to:")
    print(REPORT_FILE)

    # --------------------------------------------------------
    # Interactive menu
    # --------------------------------------------------------

    while True:

        print("\n" + "=" * 80)
        print("RESULT DISPLAY OPTIONS")
        print("=" * 80)

        print("1. Review summary")
        print("2. Correct pairs (based on cosine threshold)")
        print("3. Correct pairs (based on paired caption ranking)")
        print("4. Incorrect pairs (based on cosine threshold)")
        print("5. Incorrect pairs (based on paired caption ranking)")
        print("6. All pairs (based on cosine threshold)")
        print("7. All pairs (based on paired caption ranking)")
        print("8. Exit")

        choice = input("\nSelect an option (1-8): ").strip()

        # ----------------------------------------------------
        # Option 1: Summary
        # ----------------------------------------------------

        if choice == "1":

            print_summary(
                total_pairs,
                cosine_correct,
                cosine_incorrect,
                paired_correct,
                paired_incorrect,
                average_paired_similarity,
                average_highest_similarity
            )

        # ----------------------------------------------------
        # Option 2: Correct by cosine threshold
        # ----------------------------------------------------

        elif choice == "2":

            selected = results_df[
                results_df["cosine_status"] == True
            ]

            print(
                f"\n{len(selected)} pairs meet the cosine threshold "
                f"of {COSINE_THRESHOLD:.4f}."
            )

        # ----------------------------------------------------
        # Option 3: Correct by paired caption ranking
        # ----------------------------------------------------

        elif choice == "3":

            selected = results_df[
                results_df["paired_status"] == True
            ]

            print(
                f"\n{len(selected)} images have their original paired "
                f"caption ranked highest."
            )

        # ----------------------------------------------------
        # Option 4: Incorrect by cosine threshold
        # ----------------------------------------------------

        elif choice == "4":

            selected = results_df[
                results_df["cosine_status"] == False
            ]

            print(
                f"\n{len(selected)} pairs are below the cosine threshold "
                f"of {COSINE_THRESHOLD:.4f}."
            )

        # ----------------------------------------------------
        # Option 5: Incorrect by paired caption ranking
        # ----------------------------------------------------

        elif choice == "5":

            selected = results_df[
                results_df["paired_status"] == False
            ]

            print(
                f"\n{len(selected)} images have another caption ranked "
                f"higher than their original paired caption."
            )

        # ----------------------------------------------------
        # Option 6: All by cosine threshold
        # ----------------------------------------------------

        elif choice == "6":

            selected = results_df

            print(f"\nDisplaying all {len(selected)} pairs.")

        # ----------------------------------------------------
        # Option 7: All by paired caption ranking
        # ----------------------------------------------------

        elif choice == "7":

            selected = results_df

            print(f"\nDisplaying all {len(selected)} pairs.")

        # ----------------------------------------------------
        # Option 8: Exit
        # ----------------------------------------------------

        elif choice == "8":

            print("\nExiting.")
            break

        else:

            print("\nInvalid selection. Please enter a number from 1 to 8.")
            continue

        # ----------------------------------------------------
        # Display selected individual results
        # ----------------------------------------------------

        if choice in ["2", "3", "4", "5", "6", "7"]:

            if len(selected) == 0:
                print("\nNo results to display.")
                continue

            for _, row in selected.iterrows():
                print_result(row)

            input("\nPress Enter to return to the menu...")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()