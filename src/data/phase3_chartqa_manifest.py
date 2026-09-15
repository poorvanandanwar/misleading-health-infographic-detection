from pathlib import Path
import json
import pandas as pd
import re
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHARTQA_ROOT = ROOT / "data" / "raw" / "chartqa" / "ChartQA Dataset"

OUTPUT_DIR = ROOT / "data" / "processed" / "phase3" / "chartqa"
INTERIM_DIR = ROOT / "data" / "interim" / "phase3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# FIND JSON FILE
# ============================================================

def find_json(split, question_type):

    filename = f"{split}_{question_type}.json"

    matches = list(CHARTQA_ROOT.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {CHARTQA_ROOT}"
        )

    if len(matches) > 1:
        print(f"WARNING: Multiple {filename} files found:")
        for m in matches:
            print("   ", m)

    return matches[0]


# ============================================================
# BUILD FILE INDEXES
# ============================================================

def build_file_indexes():

    print("\n" + "=" * 70)
    print("BUILDING CHARTQA FILE INDEXES")
    print("=" * 70)

    image_index = {}
    table_index = {}

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    print("\nScanning PNG images...")

    start = time.time()

    for path in CHARTQA_ROOT.rglob("*.png"):

        filename = path.name

        # Store using filename as key.
        # ChartQA image names are unique.
        if filename not in image_index:
            image_index[filename] = path

    elapsed = time.time() - start

    print(f"Images indexed: {len(image_index)}")
    print(f"Time: {elapsed:.2f} seconds")

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    print("\nScanning CSV tables...")

    start = time.time()

    for path in CHARTQA_ROOT.rglob("*.csv"):

        filename = path.name

        if filename not in table_index:
            table_index[filename] = path

    elapsed = time.time() - start

    print(f"Tables indexed: {len(table_index)}")
    print(f"Time: {elapsed:.2f} seconds")

    return image_index, table_index


# ============================================================
# PROCESS ONE JSON FILE
# ============================================================

def process_json(
    json_path,
    split,
    question_type,
    image_index,
    table_index,
    global_counter
):

    print("\n" + "-" * 70)
    print(f"{split.upper()} - {question_type.upper()}")
    print("-" * 70)

    print("JSON:", json_path)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("JSON records:", len(records))

    rows = []

    missing_images = 0
    missing_tables = 0

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    for i, record in enumerate(records):

        image_name = clean_text(record.get("imgname", ""))

        question = clean_text(record.get("query", ""))

        answer = clean_text(record.get("label", ""))

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_path = image_index.get(image_name)

        image_exists = image_path is not None

        if not image_exists:
            missing_images += 1

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        table_name = Path(image_name).stem + ".csv"

        table_path = table_index.get(table_name)

        table_exists = table_path is not None

        if not table_exists:
            missing_tables += 1

        # ----------------------------------------------------
        # Sample ID
        # ----------------------------------------------------

        sample_id = (
            f"chartqa_{split}_{question_type}_{global_counter[0]:07d}"
        )

        global_counter[0] += 1

        # ----------------------------------------------------
        # Row
        # ----------------------------------------------------

        rows.append({
            "sample_id": sample_id,
            "dataset": "ChartQA",
            "split": split,
            "question_type": question_type,

            "image_name": image_name,
            "image_path": str(image_path) if image_path else "",
            "image_exists": image_exists,

            "table_name": table_name,
            "table_path": str(table_path) if table_path else "",
            "table_exists": table_exists,

            "question": question,
            "answer": answer
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 5000 == 0:
            print(
                f"Processed {i + 1:,} / {len(records):,} records"
            )

    print("Valid records:", len(rows))
    print("Missing images:", missing_images)
    print("Missing tables:", missing_tables)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3A - CHARTQA QUESTION MANIFEST")
    print("=" * 70)

    start_total = time.time()

    # --------------------------------------------------------
    # Build indexes ONCE
    # --------------------------------------------------------

    image_index, table_index = build_file_indexes()

    # --------------------------------------------------------
    # Process all six JSON files
    # --------------------------------------------------------

    splits = ["train", "val", "test"]
    question_types = ["human", "augmented"]

    all_rows = []

    global_counter = [0]

    for split in splits:

        for question_type in question_types:

            json_path = find_json(split, question_type)

            rows = process_json(
                json_path=json_path,
                split=split,
                question_type=question_type,
                image_index=image_index,
                table_index=table_index,
                global_counter=global_counter
            )

            all_rows.extend(rows)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING QUESTION MANIFEST")
    print("=" * 70)

    df = pd.DataFrame(all_rows)

    # --------------------------------------------------------
    # Save main manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_DIR / "chartqa_question_manifest.csv"
    )

    df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(manifest_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "metric": [
            "total_questions",
            "total_images_indexed",
            "total_tables_indexed",
            "missing_images",
            "missing_tables"
        ],
        "value": [
            len(df),
            len(image_index),
            len(table_index),
            int((~df["image_exists"]).sum()),
            int((~df["table_exists"]).sum())
        ]
    })

    summary_path = (
        INTERIM_DIR / "chartqa_question_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nSummary saved:")
    print(summary_path)

    # --------------------------------------------------------
    # Split / type summary
    # --------------------------------------------------------

    breakdown = (
        df.groupby(
            ["split", "question_type"]
        )
        .size()
        .reset_index(name="question_count")
    )

    breakdown_path = (
        INTERIM_DIR / "chartqa_question_breakdown.csv"
    )

    breakdown.to_csv(
        breakdown_path,
        index=False
    )

    print("\nBreakdown:")
    print(breakdown.to_string(index=False))

    # --------------------------------------------------------
    # Missing assets
    # --------------------------------------------------------

    missing_df = df[
        (~df["image_exists"]) |
        (~df["table_exists"])
    ].copy()

    missing_path = (
        INTERIM_DIR / "chartqa_missing_assets.csv"
    )

    missing_df.to_csv(
        missing_path,
        index=False
    )

    print("\nMissing asset rows:", len(missing_df))

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3A COMPLETE")
    print("=" * 70)

    print(f"\nTotal question records: {len(df):,}")

    print(
        f"Questions with images: "
        f"{df['image_exists'].sum():,}"
    )

    print(
        f"Questions with tables: "
        f"{df['table_exists'].sum():,}"
    )

    print(
        f"Missing images: "
        f"{(~df['image_exists']).sum():,}"
    )

    print(
        f"Missing tables: "
        f"{(~df['table_exists']).sum():,}"
    )

    print(
        f"\nTotal time: "
        f"{time.time() - start_total:.2f} seconds"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()