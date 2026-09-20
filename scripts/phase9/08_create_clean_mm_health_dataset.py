import pandas as pd
import os
import json

from sklearn.model_selection import train_test_split

INPUT_PATH = (
    "outputs/phase9/mm_health_analysis/"
    "mm_health_metadata.csv"
)

OUTPUT_DIR = (
    "outputs/phase9/mm_health_analysis/"
    "clean_dataset"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MM-Health metadata...")

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Create unique record key
df["record_key"] = (
    df["subset"].astype(str)
    + "_"
    + df["id"].astype(str)
)

# Remove duplicate records
df_clean = df.drop_duplicates(
    subset="record_key",
    keep="first"
).copy()

print(
    "\nShape after duplicate removal:",
    df_clean.shape
)

print("\nLabel distribution:")
print(
    df_clean["label"].value_counts(
        dropna=False
    )
)

print("\nUnique labels:")
print(df_clean["label"].unique())

# Create train and temporary sets
train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean["label"]
)

# Create validation and test sets
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Assign new split names
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["new_split"] = "train"
val_df["new_split"] = "validation"
test_df["new_split"] = "test"

# Save datasets
train_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_train_clean.csv"
)

val_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_validation_clean.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_test_clean.csv"
)

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# Check overlap
train_keys = set(train_df["record_key"])
val_keys = set(val_df["record_key"])
test_keys = set(test_df["record_key"])

train_val_overlap = train_keys.intersection(
    val_keys
)

train_test_overlap = train_keys.intersection(
    test_keys
)

val_test_overlap = val_keys.intersection(
    test_keys
)

print("\nNew split sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nNew split overlaps:")
print(
    "Train-validation:",
    len(train_val_overlap)
)

print(
    "Train-test:",
    len(train_test_overlap)
)

print(
    "Validation-test:",
    len(val_test_overlap)
)

# Save summary
summary = {
    "original_records": int(len(df)),
    "unique_records": int(len(df_clean)),
    "train_records": int(len(train_df)),
    "validation_records": int(len(val_df)),
    "test_records": int(len(test_df)),
    "train_validation_overlap": int(
        len(train_val_overlap)
    ),
    "train_test_overlap": int(
        len(train_test_overlap)
    ),
    "validation_test_overlap": int(
        len(val_test_overlap)
    ),
    "label_distribution": df_clean[
        "label"
    ].value_counts(
        dropna=False
    ).to_dict()
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "clean_mm_health_dataset_summary.json"
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

print("\nSaved files:")
print(train_path)
print(val_path)
print(test_path)
print(summary_path)

print("\nPhase 9 clean dataset creation completed.")