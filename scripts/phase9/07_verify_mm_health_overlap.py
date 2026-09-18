import pandas as pd
import os
import json

INPUT_PATH = (
    "outputs/phase9/mm_health_analysis/"
    "mm_health_metadata.csv"
)

OUTPUT_DIR = "outputs/phase9/mm_health_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MM-Health metadata...")

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

# Check exact duplicate rows
exact_duplicates = df.duplicated().sum()

print("\nExact duplicate rows:", exact_duplicates)

# Create unique record key
df["record_key"] = (
    df["subset"].astype(str)
    + "_"
    + df["id"].astype(str)
)

key_counts = df["record_key"].value_counts()

duplicate_keys = key_counts[
    key_counts > 1
].index

print(
    "Duplicate subset-ID keys:",
    len(duplicate_keys)
)

# Separate train and test
train_df = df[
    df["split"] == "train"
].copy()

test_df = df[
    df["split"] == "test"
].copy()

train_keys = set(train_df["record_key"])

test_keys = set(test_df["record_key"])

overlap_keys = train_keys.intersection(
    test_keys
)

print("\nTrain unique keys:", len(train_keys))

print("Test unique keys:", len(test_keys))

print("Overlapping keys:", len(overlap_keys))

# Extract overlapping records
overlap_df = df[
    df["record_key"].isin(overlap_keys)
].copy()

# Compare image paths and labels
comparison = (
    overlap_df
    .groupby("record_key")
    .agg(
        split_count=("split", "nunique"),
        image_count=("original_image", "nunique"),
        label_count=("label", "nunique"),
        subset_count=("subset", "nunique")
    )
    .reset_index()
)

print("\nOverlap comparison:")
print(comparison.head(10))

print("\nImage path count:")
print(
    comparison["image_count"].value_counts()
)

print("\nLabel count:")
print(
    comparison["label_count"].value_counts()
)

# Save overlap records
overlap_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_train_test_overlap.csv"
)

overlap_df.to_csv(
    overlap_path,
    index=False
)

# Save comparison report
comparison_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_overlap_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

# Generate summary
summary = {
    "total_records": int(len(df)),
    "exact_duplicate_rows": int(
        exact_duplicates
    ),
    "train_unique_keys": int(
        len(train_keys)
    ),
    "test_unique_keys": int(
        len(test_keys)
    ),
    "overlapping_keys": int(
        len(overlap_keys)
    ),
    "overlap_image_count_distribution": (
        comparison["image_count"]
        .value_counts()
        .to_dict()
    ),
    "overlap_label_count_distribution": (
        comparison["label_count"]
        .value_counts()
        .to_dict()
    )
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_overlap_verification_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        default=str
    )

print("\nReports saved:")
print(overlap_path)
print(comparison_path)
print(summary_path)

print("\nMM-Health overlap verification completed.")