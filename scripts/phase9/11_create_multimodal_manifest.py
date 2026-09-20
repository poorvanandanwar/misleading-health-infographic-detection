from datasets import load_dataset
import pandas as pd
import os
import json
import ast

DATASET_NAME = "zzha6204/MM-Health"

BASE_DIR = "outputs/phase9/mm_health_analysis"
CLEAN_DIR = os.path.join(BASE_DIR, "clean_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "multimodal_manifest")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSETS = [
    "Med-MMHL",
    "MM-COVID19",
    "ReCOVery",
    "MMCoVar"
]

print("Loading clean split files...")

train_df = pd.read_csv(
    os.path.join(CLEAN_DIR, "mm_health_train_clean.csv")
)

val_df = pd.read_csv(
    os.path.join(CLEAN_DIR, "mm_health_validation_clean.csv")
)

test_df = pd.read_csv(
    os.path.join(CLEAN_DIR, "mm_health_test_clean.csv")
)

train_keys = set(train_df["record_key"].astype(str))
val_keys = set(val_df["record_key"].astype(str))
test_keys = set(test_df["record_key"].astype(str))

print("Train keys:", len(train_keys))
print("Validation keys:", len(val_keys))
print("Test keys:", len(test_keys))

print("\nLoading MM-Health dataset...")

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

                record_id = str(item.get("id"))

                record_key = (
                    subset_name + "_" + record_id
                )

                image_data = item.get("image", {})
                text_data = item.get("text", {})

                if not isinstance(image_data, dict):
                    continue

                if not isinstance(text_data, dict):
                    continue

                original_images = image_data.get(
                    "original",
                    []
                )

                original_text = text_data.get(
                    "original",
                    ""
                )

                if isinstance(original_images, str):
                    original_images = [original_images]

                if isinstance(original_text, list):
                    original_text = "\n".join(
                        str(x) for x in original_text
                    )

                if original_text is None:
                    original_text = ""

                original_text = str(original_text).strip()

                if not isinstance(original_images, list):
                    original_images = []

                original_images = [
                    str(image).strip()
                    for image in original_images
                    if str(image).strip()
                ]

                if len(original_images) == 0:
                    continue

                if len(original_text) == 0:
                    continue

                records.append({
                    "record_key": record_key,
                    "subset": subset_name,
                    "record_id": record_id,
                    "label": item.get("label"),
                    "source": item.get("source"),
                    "is_english": item.get("is_english"),
                    "image_paths": json.dumps(
                        original_images
                    ),
                    "image_count": len(original_images),
                    "text": original_text
                })

print("\nTotal extracted records:", len(records))

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No valid multimodal records were extracted."
    )

print("\nRemoving duplicate record keys...")

df = df.drop_duplicates(
    subset=["record_key"],
    keep="first"
)

print("Records after deduplication:", len(df))

print("\nChecking duplicate image references...")

image_rows = []

for _, row in df.iterrows():

    images = json.loads(row["image_paths"])

    for image_path in images:

        image_rows.append({
            "record_key": row["record_key"],
            "image_path": image_path
        })

image_df = pd.DataFrame(image_rows)

duplicate_images = image_df[
    image_df["image_path"].duplicated(
        keep=False
    )
].sort_values("image_path")

duplicate_image_count = (
    duplicate_images["image_path"]
    .nunique()
)

print(
    "Duplicate image paths:",
    duplicate_image_count
)

duplicate_images.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "duplicate_image_references.csv"
    ),
    index=False
)

print("\nAssigning clean splits...")

def assign_split(record_key):

    if record_key in train_keys:
        return "train"

    if record_key in val_keys:
        return "validation"

    if record_key in test_keys:
        return "test"

    return None


df["split"] = df["record_key"].apply(
    assign_split
)

df = df[
    df["split"].notna()
].copy()

print("\nFinal split distribution:")
print(df["split"].value_counts())

print("\nFinal label distribution:")
print(df["label"].value_counts())

print("\nSaving complete manifest...")

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multimodal_manifest.csv"
    ),
    index=False
)

for split_name in [
    "train",
    "validation",
    "test"
]:

    split_df = df[
        df["split"] == split_name
    ].copy()

    split_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"multimodal_{split_name}.csv"
        ),
        index=False
    )

summary = {
    "total_records": int(len(df)),
    "train_records": int(
        (df["split"] == "train").sum()
    ),
    "validation_records": int(
        (df["split"] == "validation").sum()
    ),
    "test_records": int(
        (df["split"] == "test").sum()
    ),
    "label_distribution": {
        str(k): int(v)
        for k, v in df["label"].value_counts().items()
    },
    "duplicate_image_paths": int(
        duplicate_image_count
    ),
    "empty_text_records": int(
        (df["text"].str.len() == 0).sum()
    )
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "multimodal_manifest_summary.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print("\nManifest creation completed.")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")
print("- multimodal_manifest.csv")
print("- multimodal_train.csv")
print("- multimodal_validation.csv")
print("- multimodal_test.csv")
print("- duplicate_image_references.csv")
print("- multimodal_manifest_summary.json")