from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE3_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase3"
)


# ============================================================
# Expected files
# ============================================================

FILES = {
    "ChartQA question manifest":
        PHASE3_DIR
        / "chartqa"
        / "chartqa_question_manifest.csv",

    "PubHealth text processed":
        PHASE3_DIR
        / "pubhealth"
        / "pubhealth_text_processed.csv",

    "HealthFC text processed":
        PHASE3_DIR
        / "healthfc"
        / "healthfc_text_processed.csv",

    "ChartQA OCR pilot":
        PHASE3_DIR
        / "ocr"
        / "chartqa_ocr_pilot.csv",

    "OCR pilot summary":
        INTERIM_DIR
        / "ocr_pilot_summary.csv",

    "Text preprocessing summary":
        INTERIM_DIR
        / "text_preprocessing_summary.csv"
}


# ============================================================
# Helper
# ============================================================

def check_file(name, path):

    if path.exists():

        size_mb = path.stat().st_size / (
            1024 * 1024
        )

        print(
            f"[OK]   {name}"
        )

        print(
            f"       {path}"
        )

        print(
            f"       Size: {size_mb:.2f} MB"
        )

        return True

    else:

        print(
            f"[FAIL] {name}"
        )

        print(
            f"       Missing: {path}"
        )

        return False


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 - FINAL VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    print()
    print("Checking expected files...")

    file_status = []

    for name, path in FILES.items():

        file_status.append(
            check_file(name, path)
        )

    # --------------------------------------------------------
    # ChartQA
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("ChartQA verification")
    print("-" * 70)

    chartqa_path = FILES[
        "ChartQA question manifest"
    ]

    if chartqa_path.exists():

        chartqa = pd.read_csv(
            chartqa_path
        )

        print()
        print(
            "Rows:",
            len(chartqa)
        )

        print()
        print("Missing images:")

        print(
            (~chartqa["image_exists"]).sum()
        )

        print()
        print("Missing tables:")

        print(
            (~chartqa["table_exists"]).sum()
        )

        print()
        print("Split distribution:")

        print(
            chartqa["split"].value_counts()
        )

        print()
        print("Question type:")

        print(
            chartqa["question_type"].value_counts()
        )

    # --------------------------------------------------------
    # PubHealth
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("PubHealth verification")
    print("-" * 70)

    pubhealth_path = FILES[
        "PubHealth text processed"
    ]

    if pubhealth_path.exists():

        pubhealth = pd.read_csv(
            pubhealth_path
        )

        print()
        print(
            "Rows:",
            len(pubhealth)
        )

        print()
        print("Empty claims:")

        print(
            pubhealth["claim_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print()
        print("Labels:")

        print(
            pubhealth["label"].value_counts()
        )

    # --------------------------------------------------------
    # HealthFC
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("HealthFC verification")
    print("-" * 70)

    healthfc_path = FILES[
        "HealthFC text processed"
    ]

    if healthfc_path.exists():

        healthfc = pd.read_csv(
            healthfc_path
        )

        print()
        print(
            "Rows:",
            len(healthfc)
        )

        print()
        print("Empty English claims:")

        print(
            healthfc["en_claim_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print()
        print("Labels:")

        print(
            healthfc["label"].value_counts()
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("OCR verification")
    print("-" * 70)

    ocr_path = FILES[
        "ChartQA OCR pilot"
    ]

    ocr_pass = False

    if ocr_path.exists():

        ocr = pd.read_csv(
            ocr_path
        )

        successful = (
            ocr["status"] == "success"
        ).sum()

        empty = (
            ocr["status"] == "empty"
        ).sum()

        errors = (
            ocr["status"] == "error"
        ).sum()

        print()
        print(
            "OCR records:",
            len(ocr)
        )

        print()
        print("Status:")

        print(
            ocr["status"].value_counts()
        )

        print()
        print(
            "Successful:",
            successful
        )

        print(
            "Empty:",
            empty
        )

        print(
            "Errors:",
            errors
        )

        # OCR is considered valid only if
        # at least one image succeeds and
        # there are no inference errors.
        if successful > 0 and errors == 0:

            ocr_pass = True

            print()
            print(
                "OCR STATUS: PASS"
            )

        else:

            print()
            print(
                "OCR STATUS: FAIL"
            )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print()
    print("=" * 70)

    all_files_exist = all(
        file_status
    )

    if all_files_exist and ocr_pass:

        print(
            "PHASE 3 COMPLETE"
        )

        print(
            "All Phase 3 preprocessing and OCR checks passed."
        )

    elif all_files_exist:

        print(
            "PHASE 3 NOT COMPLETE"
        )

        print(
            "All files exist, but OCR verification has not passed."
        )

    else:

        print(
            "PHASE 3 NOT COMPLETE"
        )

        print(
            "One or more required files are missing."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()