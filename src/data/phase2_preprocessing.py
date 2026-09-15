from pathlib import Path
import pandas as pd
import numpy as np
import json
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "phase2"
INTERIM_DIR = ROOT / "data" / "interim" / "phase2"

PUBHEALTH_RAW = RAW_DIR / "pubhealth"
HEALTHFC_RAW = RAW_DIR / "healthfc"
CHARTQA_RAW = RAW_DIR / "chartqa" / "ChartQA Dataset"

PUBHEALTH_OUT = PROCESSED_DIR / "pubhealth"
HEALTHFC_OUT = PROCESSED_DIR / "healthfc"
CHARTQA_OUT = PROCESSED_DIR / "chartqa"


for directory in [
    PROCESSED_DIR,
    INTERIM_DIR,
    PUBHEALTH_OUT,
    HEALTHFC_OUT,
    CHARTQA_OUT
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Basic text cleaning.
    Does not change meaning.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Normalize whitespace
    value = " ".join(value.split())

    return value.strip()


def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# PUBHEALTH
# ============================================================

def process_pubhealth():

    print_separator("PHASE 2 - PUBHEALTH CLEANING")

    split_files = {
        "train": PUBHEALTH_RAW / "train.tsv",
        "dev": PUBHEALTH_RAW / "dev.tsv",
        "test": PUBHEALTH_RAW / "test.tsv"
    }

    valid_labels = {
        "true",
        "false",
        "mixture",
        "unproven"
    }

    processed_frames = []

    for split, filepath in split_files.items():

        print(f"\nProcessing {split}...")
        print(f"File: {filepath}")

        df = pd.read_csv(filepath, sep="\t")

        print("Original shape:", df.shape)

        # ----------------------------------------------------
        # Remove accidental unnamed/index columns
        # ----------------------------------------------------

        unnamed_columns = [
            col for col in df.columns
            if str(col).lower().startswith("unnamed")
        ]

        if unnamed_columns:
            print("Removing columns:", unnamed_columns)
            df = df.drop(columns=unnamed_columns)

        # ----------------------------------------------------
        # Keep expected columns when available
        # ----------------------------------------------------

        expected_columns = [
            "claim_id",
            "claim",
            "date_published",
            "explanation",
            "fact_checkers",
            "main_text",
            "sources",
            "label",
            "subjects"
        ]

        available_columns = [
            col for col in expected_columns
            if col in df.columns
        ]

        df = df[available_columns]

        # ----------------------------------------------------
        # Record original row count
        # ----------------------------------------------------

        original_count = len(df)

        # ----------------------------------------------------
        # Clean label
        # ----------------------------------------------------

        if "label" in df.columns:
            df["label"] = (
                df["label"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

        # ----------------------------------------------------
        # Remove invalid/missing labels
        # ----------------------------------------------------

        invalid_label_mask = ~df["label"].isin(valid_labels)

        invalid_count = invalid_label_mask.sum()

        print("Invalid/missing labels removed:", invalid_count)

        df = df[~invalid_label_mask].copy()

        # ----------------------------------------------------
        # Remove missing claims
        # ----------------------------------------------------

        before_claim_filter = len(df)

        df["claim"] = df["claim"].apply(clean_text)

        df = df[df["claim"] != ""].copy()

        missing_claims_removed = (
            before_claim_filter - len(df)
        )

        print(
            "Missing/empty claims removed:",
            missing_claims_removed
        )

        # ----------------------------------------------------
        # Clean text columns
        # ----------------------------------------------------

        text_columns = [
            "claim",
            "explanation",
            "fact_checkers",
            "main_text",
            "sources",
            "subjects"
        ]

        for column in text_columns:

            if column in df.columns:
                df[column] = df[column].apply(clean_text)

        # ----------------------------------------------------
        # Remove exact duplicates
        # ----------------------------------------------------

        before_duplicates = len(df)

        df = df.drop_duplicates().copy()

        duplicates_removed = (
            before_duplicates - len(df)
        )

        print("Duplicate rows removed:", duplicates_removed)

        # ----------------------------------------------------
        # Add split
        # ----------------------------------------------------

        df["split"] = split

        # ----------------------------------------------------
        # Add processed sample ID
        # ----------------------------------------------------

        df["sample_id"] = [
            f"pubhealth_{split}_{i:06d}"
            for i in range(len(df))
        ]

        # ----------------------------------------------------
        # Reorder columns
        # ----------------------------------------------------

        preferred_columns = [
            "sample_id",
            "claim_id",
            "claim",
            "explanation",
            "main_text",
            "sources",
            "subjects",
            "fact_checkers",
            "date_published",
            "label",
            "split"
        ]

        preferred_columns = [
            col for col in preferred_columns
            if col in df.columns
        ]

        df = df[preferred_columns]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = PUBHEALTH_OUT / f"{split}.csv"

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        print("Cleaned shape:", df.shape)
        print("Saved:", output_file)

        processed_frames.append(df)

    # --------------------------------------------------------
    # Combine all splits
    # --------------------------------------------------------

    combined = pd.concat(
        processed_frames,
        ignore_index=True
    )

    combined.to_csv(
        PUBHEALTH_OUT / "pubhealth_clean.csv",
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save label distribution
    # --------------------------------------------------------

    label_counts = (
        combined["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    label_counts.to_csv(
        INTERIM_DIR / "pubhealth_clean_label_counts.csv",
        index=False
    )

    print("\nFinal PubHealth shape:", combined.shape)
    print("\nFinal labels:")
    print(label_counts)

    print(
        "\nSaved combined file:",
        PUBHEALTH_OUT / "pubhealth_clean.csv"
    )


# ============================================================
# HEALTHFC
# ============================================================

def process_healthfc():

    print_separator("PHASE 2 - HEALTHFC CLEANING")

    input_file = HEALTHFC_RAW / "Datensatz.csv"

    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates().copy()

    print(
        "Duplicate rows removed:",
        before_duplicates - len(df)
    )

    # --------------------------------------------------------
    # Remove rows with missing claims
    # --------------------------------------------------------

    df["en_claim"] = df["en_claim"].apply(clean_text)

    before_claim_filter = len(df)

    df = df[df["en_claim"] != ""].copy()

    print(
        "Missing/empty claims removed:",
        before_claim_filter - len(df)
    )

    # --------------------------------------------------------
    # Clean text fields
    # --------------------------------------------------------

    text_columns = [
        "en_claim",
        "en_explanation",
        "en_top_sentences",
        "de_claim",
        "de_explanation",
        "de_top_sentences",
        "de_verdict",
        "de_title",
        "authors"
    ]

    for column in text_columns:

        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    # --------------------------------------------------------
    # Keep original label
    # --------------------------------------------------------

    df["original_label"] = df["label"]

    # --------------------------------------------------------
    # Split dataset
    #
    # HealthFC does not provide train/val/test in this CSV,
    # so we do NOT create a split here.
    # --------------------------------------------------------

    df["split"] = "unsplit"

    # --------------------------------------------------------
    # Sample IDs
    # --------------------------------------------------------

    df["sample_id"] = [
        f"healthfc_{i:06d}"
        for i in range(len(df))
    ]

    # --------------------------------------------------------
    # Reorder
    # --------------------------------------------------------

    preferred_columns = [
        "sample_id",
        "en_claim",
        "en_explanation",
        "en_top_sentences",
        "de_claim",
        "de_explanation",
        "de_top_sentences",
        "original_label",
        "label",
        "de_verdict",
        "de_title",
        "authors",
        "date",
        "url",
        "split"
    ]

    preferred_columns = [
        col for col in preferred_columns
        if col in df.columns
    ]

    df = df[preferred_columns]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = HEALTHFC_OUT / "healthfc_clean.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    label_counts = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    label_counts.to_csv(
        INTERIM_DIR / "healthfc_clean_label_counts.csv",
        index=False
    )

    print("\nFinal HealthFC shape:", df.shape)

    print("\nLabels:")
    print(label_counts)

    print("\nSaved:", output_file)


# ============================================================
# CHARTQA
# ============================================================

def process_chartqa():

    print_separator("PHASE 2 - CHARTQA VALIDATION")

    records = []

    valid_image_count = 0
    invalid_image_count = 0

    csv_count = 0
    json_count = 0

    split_directories = [
        ("train", CHARTQA_RAW / "train"),
        ("val", CHARTQA_RAW / "val"),
        ("test", CHARTQA_RAW / "test")
    ]

    for split, split_dir in split_directories:

        print(f"\nChecking {split}...")

        if not split_dir.exists():
            print("Missing split directory:", split_dir)
            continue

        # ----------------------------------------------------
        # PNG files
        # ----------------------------------------------------

        png_files = list(split_dir.rglob("*.png"))

        print("PNG files:", len(png_files))

        for image_path in png_files:

            relative_path = image_path.relative_to(
                CHARTQA_RAW
            )

            image_valid = True
            width = None
            height = None

            try:

                with Image.open(image_path) as img:

                    img.verify()

                # Reopen after verify
                with Image.open(image_path) as img:

                    width, height = img.size

            except Exception as e:

                image_valid = False
                invalid_image_count += 1

                print(
                    "Invalid image:",
                    image_path,
                    "|",
                    str(e)
                )

            if image_valid:
                valid_image_count += 1

            records.append({
                "sample_id": image_path.stem,
                "split": split,
                "file_type": "image",
                "file_path": str(relative_path),
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "valid": image_valid
            })

        # ----------------------------------------------------
        # CSV files
        # ----------------------------------------------------

        csv_files = list(split_dir.rglob("*.csv"))

        print("CSV files:", len(csv_files))

        for csv_path in csv_files:

            csv_count += 1

            relative_path = csv_path.relative_to(
                CHARTQA_RAW
            )

            valid_table = True
            rows = None
            columns = None

            try:

                table = pd.read_csv(csv_path)

                rows, columns = table.shape

                if rows == 0:
                    valid_table = False

            except Exception as e:

                valid_table = False

                print(
                    "Invalid CSV:",
                    csv_path,
                    "|",
                    str(e)
                )

            records.append({
                "sample_id": csv_path.stem,
                "split": split,
                "file_type": "table",
                "file_path": str(relative_path),
                "file_name": csv_path.name,
                "width": None,
                "height": None,
                "valid": valid_table,
                "table_rows": rows,
                "table_columns": columns
            })

        # ----------------------------------------------------
        # JSON files
        # ----------------------------------------------------

        json_files = list(split_dir.rglob("*.json"))

        print("JSON files:", len(json_files))

        for json_path in json_files:

            json_count += 1

            relative_path = json_path.relative_to(
                CHARTQA_RAW
            )

            json_valid = True

            try:

                with open(
                    json_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    json.load(f)

            except Exception as e:

                json_valid = False

                print(
                    "Invalid JSON:",
                    json_path,
                    "|",
                    str(e)
                )

            records.append({
                "sample_id": json_path.stem,
                "split": split,
                "file_type": "json",
                "file_path": str(relative_path),
                "file_name": json_path.name,
                "width": None,
                "height": None,
                "valid": json_valid
            })

    # --------------------------------------------------------
    # Save validation manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(records)

    manifest_file = (
        CHARTQA_OUT / "chartqa_file_manifest.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "metric": "valid_images",
            "value": valid_image_count
        },
        {
            "metric": "invalid_images",
            "value": invalid_image_count
        },
        {
            "metric": "csv_files",
            "value": csv_count
        },
        {
            "metric": "json_files",
            "value": json_count
        },
        {
            "metric": "total_records",
            "value": len(manifest)
        }
    ])

    summary.to_csv(
        INTERIM_DIR / "chartqa_validation_summary.csv",
        index=False
    )

    print("\nChartQA validation complete.")

    print(
        "\nValid images:",
        valid_image_count
    )

    print(
        "Invalid images:",
        invalid_image_count
    )

    print(
        "CSV files:",
        csv_count
    )

    print(
        "JSON files:",
        json_count
    )

    print(
        "\nSaved:",
        manifest_file
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PHASE 2 - DATA CLEANING AND PREPROCESSING")
    print("=" * 70)

    process_pubhealth()

    process_healthfc()

    process_chartqa()

    print("\n")
    print("=" * 70)
    print("PHASE 2 COMPLETED")
    print("=" * 70)