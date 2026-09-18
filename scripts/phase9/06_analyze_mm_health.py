from datasets import load_dataset
import pandas as pd
import json
import os

DATASET_NAME = "zzha6204/MM-Health"

OUTPUT_DIR = "outputs/phase9/mm_health_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSETS = [
    "Med-MMHL",
    "MM-COVID19",
    "ReCOVery",
    "MMCoVar"
]

print("Loading MM-Health dataset...")
print("Streaming mode enabled.")

dataset = load_dataset(
    DATASET_NAME,
    streaming=True
)

records = []

for split_name, split_data in dataset.items():

    print("\nProcessing split:", split_name)

    for outer_record in split_data:

        for subset_name in SUBSETS:

            subset_records = outer_record.get(
                subset_name,
                []
            )

            if not isinstance(subset_records, list):
                continue

            for item in subset_records:

                image_data = item.get("image", {})
                text_data = item.get("text", {})

                original_image = None

                if isinstance(image_data, dict):
                    original_image = image_data.get(
                        "original"
                    )

                record = {
                    "split": split_name,
                    "subset": subset_name,
                    "id": item.get("id"),
                    "index": item.get("index"),
                    "label": item.get("label"),
                    "source": item.get("source"),
                    "is_english": item.get("is_english"),
                    "original_image": str(
                        original_image
                    ),
                    "image_keys": list(
                        image_data.keys()
                    ) if isinstance(
                        image_data, dict
                    ) else [],
                    "text_keys": list(
                        text_data.keys()
                    ) if isinstance(
                        text_data, dict
                    ) else []
                }

                records.append(record)

print("\nTotal extracted records:", len(records))

df = pd.DataFrame(records)

if df.empty:
    print("ERROR: No records extracted.")
    raise SystemExit

csv_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_metadata.csv"
)

df.to_csv(csv_path, index=False)

print("\nMetadata saved to:")
print(csv_path)

print("\nDataset shape:")
print(df.shape)

print("\nSubset distribution:")
print(df["subset"].value_counts(dropna=False))

print("\nSplit distribution:")
print(df["split"].value_counts(dropna=False))

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))

print("\nSource distribution:")
print(df["source"].value_counts(dropna=False))

print("\nEnglish-language distribution:")
print(df["is_english"].value_counts(dropna=False))

print("\nChecking duplicate records...")

df["record_key"] = (
    df["subset"].astype(str)
    + "_"
    + df["id"].astype(str)
)

duplicate_keys = df[
    df["record_key"].duplicated(keep=False)
]

print(
    "Duplicate subset-ID records:",
    len(duplicate_keys)
)

print("\nChecking cross-split overlap...")

split_counts = (
    df.groupby("record_key")["split"]
    .nunique()
)

overlapping_keys = split_counts[
    split_counts > 1
]

print(
    "Records appearing in multiple splits:",
    len(overlapping_keys)
)

summary = {
    "total_records": int(len(df)),
    "columns": df.columns.tolist(),
    "subset_distribution": df[
        "subset"
    ].value_counts(
        dropna=False
    ).to_dict(),
    "split_distribution": df[
        "split"
    ].value_counts(
        dropna=False
    ).to_dict(),
    "label_distribution": df[
        "label"
    ].value_counts(
        dropna=False
    ).to_dict(),
    "duplicate_subset_id_records": int(
        len(duplicate_keys)
    ),
    "cross_split_overlap_keys": int(
        len(overlapping_keys)
    )
}

json_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_analysis_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        default=str
    )

print("\nSummary saved to:")
print(json_path)

print("\nPhase 9 MM-Health analysis completed.")