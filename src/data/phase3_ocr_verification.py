from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "ocr"
    / "chartqa_ocr_pilot.csv"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase3"
)

SUMMARY_PATH = (
    SUMMARY_DIR
    / "ocr_pilot_summary.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3E - OCR VERIFICATION")
    print("=" * 70)

    if not OCR_PATH.exists():

        raise FileNotFoundError(
            f"OCR pilot file not found:\n{OCR_PATH}"
        )

    df = pd.read_csv(OCR_PATH)

    print()
    print("OCR records:", len(df))

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    print()
    print("Status distribution:")

    status_counts = df["status"].value_counts()

    print(status_counts)

    successful = int(
        (df["status"] == "success").sum()
    )

    empty = int(
        (df["status"] == "empty").sum()
    )

    errors = int(
        (df["status"] == "error").sum()
    )

    print()
    print("Successful OCR:", successful)
    print("Empty OCR:", empty)
    print("Error records:", errors)

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    df["ocr_text"] = (
        df["ocr_text"]
        .fillna("")
        .astype(str)
    )

    df["char_count"] = (
        df["ocr_text"]
        .str.len()
    )

    df["word_count"] = (
        df["ocr_text"]
        .str.split()
        .str.len()
    )

    print()
    print("Average characters:")
    print(
        round(
            df["char_count"].mean(),
            2
        )
    )

    print()
    print("Average words:")
    print(
        round(
            df["word_count"].mean(),
            2
        )
    )

    print()
    print("Average text elements:")
    print(
        round(
            df["text_element_count"].mean(),
            2
        )
    )

    print()
    print("Average processing time:")
    print(
        round(
            df["processing_time_sec"].mean(),
            3
        ),
        "seconds/image"
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print()
    print("OCR examples:")

    successful_df = df[
        df["status"] == "success"
    ]

    if len(successful_df) > 0:

        print(
            successful_df[
                [
                    "image_name",
                    "ocr_text"
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

    else:

        print("No successful OCR examples available.")

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    if errors > 0:

        print()
        print("First OCR errors:")

        print(
            df[
                df["status"] == "error"
            ][
                [
                    "image_name",
                    "error"
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "total_records": len(df),
                "successful_ocr": successful,
                "empty_ocr": empty,
                "error_records": errors,
                "average_characters": df["char_count"].mean(),
                "average_words": df["word_count"].mean(),
                "average_text_elements": df["text_element_count"].mean(),
                "average_processing_time_sec": df[
                    "processing_time_sec"
                ].mean()
            }
        ]
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("Saved summary:")
    print(SUMMARY_PATH)

    print()
    print("=" * 70)

    if errors == 0 and successful > 0:

        print("OCR VERIFICATION: PASS")

    elif successful > 0:

        print("OCR VERIFICATION: PARTIAL")

    else:

        print("OCR VERIFICATION: FAIL")

    print("=" * 70)


if __name__ == "__main__":
    main()