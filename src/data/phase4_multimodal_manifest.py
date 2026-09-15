from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "claim_verification"
)

CHARTQA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "chartqa"
    / "chartqa_multimodal_pilot.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "multimodal"
)


# ============================================================
# STANDARD COLUMNS
# ============================================================

COLUMNS = [
    "sample_id",
    "source_dataset",
    "task_type",
    "split",
    "image_name",
    "image_path",
    "ocr_text",
    "claim_text",
    "evidence_text",
    "explanation",
    "question",
    "answer",
    "original_label",
    "standardized_label",
    "binary_target"
]


# ============================================================
# CLAIM DATA
# ============================================================

def load_claim_data():

    parts = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        path = (
            CLAIM_DIR
            / f"{split}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing split:\n{path}"
            )

        df = pd.read_csv(
            path
        )

        # Claim datasets do not have images
        df["image_name"] = ""
        df["image_path"] = ""
        df["ocr_text"] = ""

        # Ensure all columns exist
        for column in COLUMNS:

            if column not in df.columns:

                df[column] = ""

        parts.append(
            df[COLUMNS]
        )

    return pd.concat(
        parts,
        ignore_index=True
    )


# ============================================================
# CHARTQA
# ============================================================

def load_chartqa():

    if not CHARTQA_PATH.exists():

        raise FileNotFoundError(
            f"Missing ChartQA integration:\n"
            f"{CHARTQA_PATH}"
        )

    df = pd.read_csv(
        CHARTQA_PATH
    )

    for column in COLUMNS:

        if column not in df.columns:

            df[column] = ""

    return df[COLUMNS]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4E - MULTIMODAL MANIFEST")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    claim_df = load_claim_data()

    chartqa_df = load_chartqa()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            claim_df,
            chartqa_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Clean missing values
    # --------------------------------------------------------

    for column in COLUMNS:

        if column == "binary_target":
            continue

        combined[column] = (
            combined[column]
            .fillna("")
            .astype(str)
        )

    # --------------------------------------------------------
    # Check duplicate sample IDs
    # --------------------------------------------------------

    duplicate_count = (
        combined["sample_id"]
        .duplicated()
        .sum()
    )

    print()
    print(
        "Duplicate sample IDs:",
        duplicate_count
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate sample IDs detected."
        )

    # --------------------------------------------------------
    # Save all
    # --------------------------------------------------------

    all_path = (
        OUTPUT_DIR
        / "multimodal_manifest.csv"
    )

    combined.to_csv(
        all_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save split files
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_df = combined[
            combined["split"] == split
        ].copy()

        output_path = (
            OUTPUT_DIR
            / f"{split}.csv"
        )

        split_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        "Total records:",
        len(combined)
    )

    print()
    print(
        "Source dataset:"
    )

    print(
        combined[
            "source_dataset"
        ].value_counts()
    )

    print()
    print(
        "Task type:"
    )

    print(
        combined[
            "task_type"
        ].value_counts()
    )

    print()
    print(
        "Split:"
    )

    print(
        combined[
            "split"
        ].value_counts()
    )

    print()
    print(
        "Saved:"
    )

    print(
        all_path
    )

    print()
    print("=" * 70)
    print("PHASE 4E COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()