
from pathlib import Path
import pandas as pd
import json
import csv

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase7"
    / "phase7d_image_label_mapping"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 7D - VERIFY IMAGE-LABEL MAPPING")
print("=" * 70)

# ---------------------------------------------------------
# 1. Inspect PubHealth TSV files
# ---------------------------------------------------------

pubhealth_dir = DATA_DIR / "raw" / "pubhealth"

tsv_files = list(pubhealth_dir.glob("*.tsv"))

print(f"\nPubHealth TSV files found: {len(tsv_files)}")

mapping_results = []

for tsv_path in tsv_files:
    print(f"\nInspecting: {tsv_path.name}")

    try:
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            nrows=5
        )

        print("Columns:")
        print(list(df.columns))

        image_columns = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "image",
                    "img",
                    "file",
                    "path",
                    "url"
                ]
            )
        ]

        label_columns = [
            col for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "label",
                    "class",
                    "target",
                    "truth",
                    "verdict"
                ]
            )
        ]

        print("Image-related columns:", image_columns)
        print("Label-related columns:", label_columns)

        for column in image_columns:
            print(f"\nSample values from {column}:")
            print(df[column].head(5).tolist())

        mapping_results.append({
            "file": str(tsv_path.relative_to(BASE_DIR)),
            "columns": ", ".join(df.columns),
            "image_columns": ", ".join(image_columns),
            "label_columns": ", ".join(label_columns),
            "status": "success"
        })

    except Exception as error:
        print(f"Error: {error}")

        mapping_results.append({
            "file": str(tsv_path.relative_to(BASE_DIR)),
            "columns": "",
            "image_columns": "",
            "label_columns": "",
            "status": str(error)
        })


# ---------------------------------------------------------
# 2. Inspect ChartQA JSON files
# ---------------------------------------------------------

chartqa_dir = (
    DATA_DIR
    / "raw"
    / "chartqa"
    / "ChartQA Dataset"
)

json_files = list(chartqa_dir.rglob("*.json"))

print(f"\nChartQA JSON files found: {len(json_files)}")

chartqa_results = []

for json_path in json_files:
    try:
        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            sample = data[:3]
        else:
            sample = data

        print(f"\nInspecting: {json_path.name}")

        if isinstance(sample, list):
            for item in sample:
                if isinstance(item, dict):
                    print("Keys:", list(item.keys()))
                    break

        elif isinstance(sample, dict):
            print("Keys:", list(sample.keys()))

        chartqa_results.append({
            "file": str(json_path.relative_to(BASE_DIR)),
            "status": "success"
        })

    except Exception as error:
        chartqa_results.append({
            "file": str(json_path.relative_to(BASE_DIR)),
            "status": str(error)
        })


# ---------------------------------------------------------
# 3. Save reports
# ---------------------------------------------------------

tsv_report_path = OUTPUT_DIR / "pubhealth_mapping_report.csv"

pd.DataFrame(mapping_results).to_csv(
    tsv_report_path,
    index=False
)

json_report_path = OUTPUT_DIR / "chartqa_json_report.csv"

pd.DataFrame(chartqa_results).to_csv(
    json_report_path,
    index=False
)

summary = {
    "pubhealth_tsv_files": len(tsv_files),
    "chartqa_json_files": len(json_files),
    "pubhealth_report": str(tsv_report_path),
    "chartqa_report": str(json_report_path)
}

summary_path = OUTPUT_DIR / "phase7d_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )

print("\n" + "=" * 70)
print("PHASE 7D COMPLETED")
print("=" * 70)
print(f"PubHealth report: {tsv_report_path}")
print(f"ChartQA report: {json_report_path}")
print(f"Summary: {summary_path}")
