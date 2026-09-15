from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "claim_verification"
    / "claim_verification_all.csv"
)

PUBHEALTH_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pubhealth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "claim_verification"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase4"
    / "split_summary.csv"
)


# ============================================================
# LOAD ORIGINAL PUBHEALTH SPLITS
# ============================================================

def load_pubhealth_splits():

    print()
    print(
        "Loading original PubHealth splits..."
    )

    train_path = (
        PUBHEALTH_RAW_DIR
        / "train.tsv"
    )

    dev_path = (
        PUBHEALTH_RAW_DIR
        / "dev.tsv"
    )

    test_path = (
        PUBHEALTH_RAW_DIR
        / "test.tsv"
    )

    train = pd.read_csv(
        train_path,
        sep="\t"
    )

    dev = pd.read_csv(
        dev_path,
        sep="\t"
    )

    test = pd.read_csv(
        test_path,
        sep="\t"
    )

    split_map = {}

    for claim_id in train["claim_id"]:

        split_map[
            f"pubhealth_{str(claim_id)}"
        ] = "train"

    for claim_id in dev["claim_id"]:

        split_map[
            f"pubhealth_{str(claim_id)}"
        ] = "val"

    for claim_id in test["claim_id"]:

        split_map[
            f"pubhealth_{str(claim_id)}"
        ] = "test"

    print(
        "PubHealth split IDs:",
        len(split_map)
    )

    return split_map


# ============================================================
# HEALTHFC STRATIFIED SPLIT
# ============================================================

def split_healthfc(
    df,
    random_state=42
):

    print()
    print(
        "Creating HealthFC stratified split..."
    )

    # 70% train, 30% temporary
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[
            "standardized_label"
        ]
    )

    # 15% validation, 15% test
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=random_state,
        stratify=temp[
            "standardized_label"
        ]
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    result = pd.concat(
        [
            train,
            val,
            test
        ],
        ignore_index=True
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4B - TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify Phase 4A output
    # --------------------------------------------------------

    if not CLAIM_PATH.exists():

        raise FileNotFoundError(
            "\nPhase 4A has not been completed yet.\n\n"
            "Missing file:\n"
            f"{CLAIM_PATH}\n\n"
            "Run this first:\n"
            "python .\\src\\data\\phase4_claim_integration.py"
        )

    df = pd.read_csv(
        CLAIM_PATH
    )

    print()
    print(
        "Input rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Separate datasets
    # --------------------------------------------------------

    pubhealth = df[
        df["source_dataset"] == "PubHealth"
    ].copy()

    healthfc = df[
        df["source_dataset"] == "HealthFC"
    ].copy()

    print()
    print(
        "PubHealth rows:",
        len(pubhealth)
    )

    print(
        "HealthFC rows:",
        len(healthfc)
    )

    # --------------------------------------------------------
    # PubHealth official splits
    # --------------------------------------------------------

    split_map = (
        load_pubhealth_splits()
    )

    pubhealth["split"] = (
        pubhealth["sample_id"]
        .map(split_map)
    )

    unresolved = (
        pubhealth["split"]
        .isna()
        .sum()
    )

    print()
    print(
        "Unresolved PubHealth splits:",
        unresolved
    )

    if unresolved > 0:

        print(
            pubhealth[
                pubhealth["split"].isna()
            ][
                [
                    "sample_id",
                    "original_label"
                ]
            ]
            .head(10)
        )

        raise ValueError(
            "Some PubHealth records could not be assigned "
            "to train/val/test."
        )

    # --------------------------------------------------------
    # HealthFC stratified split
    # --------------------------------------------------------

    healthfc = split_healthfc(
        healthfc
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            pubhealth,
            healthfc
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Verify every row has a split
    # --------------------------------------------------------

    if combined["split"].isna().any():

        raise ValueError(
            "Some rows have no split."
        )

    # --------------------------------------------------------
    # Save individual splits
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SPLIT COUNTS")
    print("=" * 70)

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

        print()
        print(
            split.upper(),
            ":",
            len(split_df)
        )

        print(
            split_df[
                "source_dataset"
            ].value_counts()
        )

        print()
        print(
            "Labels:"
        )

        print(
            split_df[
                "standardized_label"
            ].value_counts()
        )

    # --------------------------------------------------------
    # Save updated complete file
    # --------------------------------------------------------

    combined.to_csv(
        CLAIM_PATH,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        combined
        .groupby(
            [
                "split",
                "source_dataset",
                "standardized_label"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print()
    print(
        "Saved split summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()
    print("=" * 70)
    print("PHASE 4B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()