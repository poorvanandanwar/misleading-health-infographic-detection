from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PUBHEALTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "pubhealth"
    / "pubhealth_text_processed.csv"
)

HEALTHFC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "healthfc"
    / "healthfc_text_processed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "claim_verification"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase4"
)


# ============================================================
# LABEL MAPPINGS
# ============================================================

# PubHealth:
# true      -> supported
# false     -> refuted
# mixture   -> mixture
# unproven  -> not enough information

PUBHEALTH_MAP = {
    "true": (
        "supported",
        0
    ),

    "false": (
        "refuted",
        1
    ),

    "mixture": (
        "mixture",
        None
    ),

    "unproven": (
        "not_enough_information",
        None
    )
}


# HealthFC:
# 0 -> Supported
# 1 -> Not enough information
# 2 -> Refuted

HEALTHFC_MAP = {
    0: (
        "supported",
        0
    ),

    1: (
        "not_enough_information",
        None
    ),

    2: (
        "refuted",
        1
    )
}


# ============================================================
# HELPER
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# PUBHEALTH
# ============================================================

def process_pubhealth():

    print()
    print("=" * 70)
    print("PROCESSING PUBHEALTH")
    print("=" * 70)

    if not PUBHEALTH_PATH.exists():

        raise FileNotFoundError(
            f"PubHealth processed file not found:\n"
            f"{PUBHEALTH_PATH}"
        )

    df = pd.read_csv(
        PUBHEALTH_PATH
    )

    print(
        "Input rows:",
        len(df)
    )

    records = []

    for _, row in df.iterrows():

        original_label = safe_text(
            row["label"]
        ).lower()

        if original_label not in PUBHEALTH_MAP:
            continue

        standardized_label, binary_target = (
            PUBHEALTH_MAP[
                original_label
            ]
        )

        # Prefer cleaned fields if available
        claim = safe_text(
            row.get(
                "claim_clean",
                row.get(
                    "claim",
                    ""
                )
            )
        )

        explanation = safe_text(
            row.get(
                "explanation_clean",
                row.get(
                    "explanation",
                    ""
                )
            )
        )

        evidence = safe_text(
            row.get(
                "main_text_clean",
                row.get(
                    "main_text",
                    ""
                )
            )
        )

        sources = safe_text(
            row.get(
                "sources",
                ""
            )
        )

        claim_id = safe_text(
            row["claim_id"]
        )

        records.append(
            {
                "sample_id":
                    f"pubhealth_{claim_id}",

                "source_dataset":
                    "PubHealth",

                "task_type":
                    "claim_verification",

                "claim_text":
                    claim,

                "evidence_text":
                    evidence,

                "explanation":
                    explanation,

                "evidence_sources":
                    sources,

                "image_name":
                    "",

                "image_path":
                    "",

                "ocr_text":
                    "",

                "question":
                    "",

                "answer":
                    "",

                "original_label":
                    original_label,

                "standardized_label":
                    standardized_label,

                "binary_target":
                    binary_target,

                "source_url":
                    ""
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        "Valid rows:",
        len(result)
    )

    return result


# ============================================================
# HEALTHFC
# ============================================================

def process_healthfc():

    print()
    print("=" * 70)
    print("PROCESSING HEALTHFC")
    print("=" * 70)

    if not HEALTHFC_PATH.exists():

        raise FileNotFoundError(
            f"HealthFC processed file not found:\n"
            f"{HEALTHFC_PATH}"
        )

    df = pd.read_csv(
        HEALTHFC_PATH
    )

    print(
        "Input rows:",
        len(df)
    )

    records = []

    for idx, row in df.iterrows():

        try:

            original_label = int(
                row["label"]
            )

        except Exception:

            continue

        if original_label not in HEALTHFC_MAP:
            continue

        standardized_label, binary_target = (
            HEALTHFC_MAP[
                original_label
            ]
        )

        claim = safe_text(
            row.get(
                "en_claim_clean",
                row.get(
                    "en_claim",
                    ""
                )
            )
        )

        explanation = safe_text(
            row.get(
                "en_explanation_clean",
                row.get(
                    "en_explanation",
                    ""
                )
            )
        )

        evidence = safe_text(
            row.get(
                "en_top_sentences_clean",
                row.get(
                    "en_top_sentences",
                    ""
                )
            )
        )

        source_url = safe_text(
            row.get(
                "url",
                ""
            )
        )

        records.append(
            {
                "sample_id":
                    f"healthfc_{idx:05d}",

                "source_dataset":
                    "HealthFC",

                "task_type":
                    "claim_verification",

                "claim_text":
                    claim,

                "evidence_text":
                    evidence,

                "explanation":
                    explanation,

                "evidence_sources":
                    "",

                "image_name":
                    "",

                "image_path":
                    "",

                "ocr_text":
                    "",

                "question":
                    "",

                "answer":
                    "",

                "original_label":
                    original_label,

                "standardized_label":
                    standardized_label,

                "binary_target":
                    binary_target,

                "source_url":
                    source_url
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        "Valid rows:",
        len(result)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4A - CLAIM DATASET INTEGRATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    pubhealth = process_pubhealth()

    healthfc = process_healthfc()

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
    # Remove duplicate sample IDs
    # --------------------------------------------------------

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["sample_id"]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Duplicates removed:",
        before - len(combined)
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTEGRATED CLAIM DATASET")
    print("=" * 70)

    print()
    print(
        "Total rows:",
        len(combined)
    )

    print()
    print(
        "Dataset distribution:"
    )

    print(
        combined[
            "source_dataset"
        ].value_counts()
    )

    print()
    print(
        "Standardized labels:"
    )

    print(
        combined[
            "standardized_label"
        ].value_counts()
    )

    print()
    print(
        "Binary target:"
    )

    print(
        combined[
            "binary_target"
        ].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "claim_verification_all.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    summary = (
        combined
        .groupby(
            [
                "source_dataset",
                "original_label",
                "standardized_label",
                "binary_target"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    summary_path = (
        INTERIM_DIR
        / "label_mapping_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        "Saved:"
    )

    print(
        summary_path
    )

    print()
    print("=" * 70)
    print("PHASE 4A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()