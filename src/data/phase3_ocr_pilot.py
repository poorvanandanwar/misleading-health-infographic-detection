import os

# ============================================================
# PaddleOCR CPU compatibility settings
# ============================================================

# Disable MKL-DNN / oneDNN because it caused the
# ConvertPirAttribute2RuntimeAttribute error on Windows.
os.environ["FLAGS_use_mkldnn"] = "0"

# ============================================================
# IMPORTANT:
# Import torch before PaddleOCR on this Windows environment.
# ============================================================

import torch

import time
from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "chartqa"
    / "chartqa_question_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase3"
    / "ocr"
)

OUTPUT_PATH = OUTPUT_DIR / "chartqa_ocr_pilot.csv"


# ============================================================
# Configuration
# ============================================================

PILOT_IMAGES = 50

CONFIDENCE_THRESHOLD = 0.30


# ============================================================
# OCR result extraction
# ============================================================

def extract_ocr_result(result, confidence_threshold=0.30):
    """
    Extract recognized text from PaddleOCR 3.x result.

    Returns:
        ocr_text
        text_element_count
    """

    texts = []
    scores = []

    try:
        # PaddleOCR 3.x result object
        if hasattr(result, "json"):
            data = result.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                import json
                data = json.loads(data)

        elif hasattr(result, "res"):
            data = result.res

        else:
            data = result

        # ----------------------------------------------------
        # Dictionary-style output
        # ----------------------------------------------------

        if isinstance(data, dict):

            # Common PaddleOCR 3.x fields
            if "rec_texts" in data:
                texts = data.get("rec_texts", []) or []

            if "rec_scores" in data:
                scores = data.get("rec_scores", []) or []

            # Sometimes nested under res
            elif "res" in data and isinstance(data["res"], dict):
                inner = data["res"]

                texts = inner.get("rec_texts", []) or []
                scores = inner.get("rec_scores", []) or []

        # ----------------------------------------------------
        # Object-style output
        # ----------------------------------------------------

        if not texts and hasattr(result, "res"):

            res = result.res

            if isinstance(res, dict):
                texts = res.get("rec_texts", []) or []
                scores = res.get("rec_scores", []) or []

        # ----------------------------------------------------
        # Apply confidence threshold
        # ----------------------------------------------------

        filtered_texts = []

        for i, text in enumerate(texts):

            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            # If score exists, apply threshold
            if i < len(scores):

                try:
                    score = float(scores[i])

                    if score < confidence_threshold:
                        continue

                except Exception:
                    pass

            filtered_texts.append(text)

        ocr_text = " ".join(filtered_texts)

        return ocr_text, len(filtered_texts)

    except Exception:
        return "", 0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3D - PADDLEOCR PILOT")
    print("=" * 70)

    print()
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    print()
    print("Loading ChartQA manifest...")

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"ChartQA manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print("Manifest rows:", len(manifest))

    # --------------------------------------------------------
    # Select unique images
    # --------------------------------------------------------

    image_df = (
        manifest[
            [
                "image_name",
                "image_path"
            ]
        ]
        .drop_duplicates(subset=["image_name"])
        .head(PILOT_IMAGES)
        .copy()
    )

    print("Unique images selected:", len(image_df))

    # --------------------------------------------------------
    # Initialize OCR
    # --------------------------------------------------------

    print()
    print("Initializing PaddleOCR...")

    ocr = PaddleOCR(
        lang="en",

        # Disable unnecessary document processing
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # Critical Windows CPU fix
        enable_mkldnn=False,

        # Keep CPU usage reasonable
        cpu_threads=4
    )

    print("PaddleOCR initialized successfully.")

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    total_start = time.time()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, row in enumerate(
        image_df.itertuples(index=False),
        start=1
    ):

        image_name = row.image_name
        image_path = row.image_path

        print()
        print(f"[{index}/{len(image_df)}] {image_name}")

        start_time = time.time()

        record = {
            "image_name": image_name,
            "image_path": image_path,
            "ocr_text": "",
            "text_element_count": 0,
            "status": "error",
            "error": "",
            "processing_time_sec": 0.0
        }

        try:

            # ------------------------------------------------
            # Validate image
            # ------------------------------------------------

            if not Path(image_path).exists():

                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            # ------------------------------------------------
            # OCR inference
            # ------------------------------------------------

            result = ocr.predict(image_path)

            # ------------------------------------------------
            # Extract text
            # ------------------------------------------------

            all_text = []
            total_elements = 0

            for res in result:

                text, count = extract_ocr_result(
                    res,
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )

                if text:
                    all_text.append(text)

                total_elements += count

            final_text = " ".join(all_text).strip()

            record["ocr_text"] = final_text
            record["text_element_count"] = total_elements

            if final_text:

                record["status"] = "success"

                print("Status: SUCCESS")
                print(
                    "Text elements:",
                    total_elements
                )

                print(
                    "OCR text:",
                    final_text[:200]
                )

            else:

                record["status"] = "empty"

                print("Status: EMPTY")

        except Exception as e:

            record["status"] = "error"

            record["error"] = str(e)

            print("Status: ERROR")
            print("Error:", str(e))

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        elapsed = time.time() - start_time

        record["processing_time_sec"] = round(
            elapsed,
            3
        )

        records.append(record)

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_df = pd.DataFrame(records)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    total_elapsed = time.time() - total_start

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OCR PILOT COMPLETE")
    print("=" * 70)

    print()
    print("Images processed:", len(output_df))

    print()
    print("Status:")

    print(
        output_df["status"].value_counts()
    )

    print()
    print(
        "Average text elements:",
        round(
            output_df["text_element_count"].mean(),
            2
        )
    )

    print(
        "Average processing time:",
        round(
            output_df["processing_time_sec"].mean(),
            2
        ),
        "seconds/image"
    )

    print(
        "Total processing time:",
        round(
            total_elapsed,
            2
        ),
        "seconds"
    )

    print()
    print("Saved:")
    print(OUTPUT_PATH)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()