from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHARTQA_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "chartqa"
    / "chartqa_question_manifest.csv"
)

OCR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "ocr"
    / "chartqa_ocr_pilot.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "chartqa"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "chartqa_multimodal_pilot.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4C - CHARTQA + OCR INTEGRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not CHARTQA_MANIFEST.exists():

        raise FileNotFoundError(
            f"ChartQA manifest not found:\n"
            f"{CHARTQA_MANIFEST}"
        )

    if not OCR_PATH.exists():

        raise FileNotFoundError(
            f"OCR pilot not found:\n"
            f"{OCR_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    chartqa = pd.read_csv(
        CHARTQA_MANIFEST
    )

    ocr = pd.read_csv(
        OCR_PATH
    )

    print()
    print(
        "ChartQA manifest rows:",
        len(chartqa)
    )

    print(
        "OCR rows:",
        len(ocr)
    )

    # --------------------------------------------------------
    # Only successful OCR
    # --------------------------------------------------------

    ocr_success = ocr[
        ocr["status"] == "success"
    ].copy()

    print(
        "Successful OCR images:",
        ocr_success[
            "image_name"
        ].nunique()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = chartqa.merge(
        ocr_success[
            [
                "image_name",
                "ocr_text",
                "text_element_count"
            ]
        ],
        on="image_name",
        how="inner"
    )

    print()
    print(
        "Integrated question records:",
        len(merged)
    )

    print(
        "Unique images:",
        merged[
            "image_name"
        ].nunique()
    )

    # --------------------------------------------------------
    # Build standardized dataset
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "sample_id":
                merged["sample_id"],

            "source_dataset":
                "ChartQA",

            "task_type":
                "chart_question_answering",

            "split":
                merged["split"],

            "image_name":
                merged["image_name"],

            "image_path":
                merged["image_path"],

            "ocr_text":
                merged["ocr_text"],

            "text_element_count":
                merged[
                    "text_element_count"
                ],

            "claim_text":
                "",

            "evidence_text":
                "",

            "explanation":
                "",

            "question":
                merged["question"],

            "answer":
                merged["answer"],

            "question_type":
                merged["question_type"],

            "original_label":
                "",

            "standardized_label":
                "",

            "binary_target":
                ""
        }
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Split distribution:"
    )

    print(
        result[
            "split"
        ].value_counts()
    )

    print()
    print(
        "Question type:"
    )

    print(
        result[
            "question_type"
        ].value_counts()
    )

    print()
    print("=" * 70)
    print("PHASE 4C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()