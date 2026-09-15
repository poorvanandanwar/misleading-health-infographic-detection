from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE4_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase4"
)


# ============================================================
# EXPECTED FILES
# ============================================================

EXPECTED_FILES = [

    PHASE4_DIR
    / "claim_verification"
    / "claim_verification_all.csv",

    PHASE4_DIR
    / "claim_verification"
    / "train.csv",

    PHASE4_DIR
    / "claim_verification"
    / "val.csv",

    PHASE4_DIR
    / "claim_verification"
    / "test.csv",

    PHASE4_DIR
    / "chartqa"
    / "chartqa_multimodal_pilot.csv",

    PHASE4_DIR
    / "multimodal"
    / "multimodal_manifest.csv",

    PHASE4_DIR
    / "multimodal"
    / "train.csv",

    PHASE4_DIR
    / "multimodal"
    / "val.csv",

    PHASE4_DIR
    / "multimodal"
    / "test.csv",

    INTERIM_DIR
    / "label_mapping_summary.csv",

    INTERIM_DIR
    / "split_summary.csv",

    INTERIM_DIR
    / "health_chart_candidates.csv"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4 - FINAL VERIFICATION")
    print("=" * 70)

    missing = []

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print()
    print("Checking expected files...")

    for path in EXPECTED_FILES:

        if path.exists():

            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"[OK] {path}"
            )

            print(
                f"     Size: {size_mb:.2f} MB"
            )

        else:

            print(
                f"[MISSING] {path}"
            )

            missing.append(
                path
            )

    # --------------------------------------------------------
    # Claim verification
    # --------------------------------------------------------

    claim_path = (
        PHASE4_DIR
        / "claim_verification"
        / "claim_verification_all.csv"
    )

    if claim_path.exists():

        claims = pd.read_csv(
            claim_path
        )

        print()
        print("=" * 70)
        print("CLAIM DATASET")
        print("=" * 70)

        print()
        print(
            "Rows:",
            len(claims)
        )

        print()
        print(
            "Sources:"
        )

        print(
            claims[
                "source_dataset"
            ].value_counts()
        )

        print()
        print(
            "Standardized labels:"
        )

        print(
            claims[
                "standardized_label"
            ].value_counts()
        )

        print()
        print(
            "Binary target:"
        )

        print(
            claims[
                "binary_target"
            ].value_counts(
                dropna=False
            )
        )

        print()
        print(
            "Missing claims:"
        )

        print(
            claims[
                "claim_text"
            ]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        )

    # --------------------------------------------------------
    # Multimodal manifest
    # --------------------------------------------------------

    mm_path = (
        PHASE4_DIR
        / "multimodal"
        / "multimodal_manifest.csv"
    )

    if mm_path.exists():

        mm = pd.read_csv(
            mm_path
        )

        print()
        print("=" * 70)
        print("MULTIMODAL MANIFEST")
        print("=" * 70)

        print()
        print(
            "Rows:",
            len(mm)
        )

        print()
        print(
            "Source datasets:"
        )

        print(
            mm[
                "source_dataset"
            ].value_counts()
        )

        print()
        print(
            "Task types:"
        )

        print(
            mm[
                "task_type"
            ].value_counts()
        )

        print()
        print(
            "Splits:"
        )

        print(
            mm[
                "split"
            ].value_counts()
        )

        print()
        print(
            "Duplicate sample IDs:"
        )

        print(
            mm[
                "sample_id"
            ]
            .duplicated()
            .sum()
        )

        # ----------------------------------------------------
        # Check ChartQA image paths
        # ----------------------------------------------------

        chart_rows = mm[
            mm["source_dataset"] == "ChartQA"
        ].copy()

        missing_images = 0

        for image_path in chart_rows[
            "image_path"
        ]:

            if not Path(
                image_path
            ).exists():

                missing_images += 1

        print()
        print(
            "ChartQA records:",
            len(chart_rows)
        )

        print(
            "Missing ChartQA images:",
            missing_images
        )

        # ----------------------------------------------------
        # OCR check
        # ----------------------------------------------------

        chart_ocr_nonempty = (
            chart_rows[
                "ocr_text"
            ]
            .fillna("")
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            "ChartQA records with OCR:",
            chart_ocr_nonempty
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if len(missing) == 0:

        print(
            "PHASE 4 COMPLETE"
        )

        print(
            "All Phase 4 integration files are present."
        )

    else:

        print(
            "PHASE 4 INCOMPLETE"
        )

        print(
            "Missing files:",
            len(missing)
        )

    print("=" * 70)


if __name__ == "__main__":
    main()