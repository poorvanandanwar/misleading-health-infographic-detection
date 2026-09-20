from pathlib import Path
import pandas as pd


# Project root
ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phase4"
    / "claim_verification"
    / "claim_verification_all.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "phase5" / "datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Original number of rows: {len(df)}")
print("Available columns:")
print(df.columns.tolist())


# Validate required columns
required_columns = [
    "sample_id",
    "split",
    "claim_text",
    "evidence_text",
    "binary_target",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Keep only binary-labeled records
df = df[df["binary_target"].notna()].copy()

df = df[
    df["binary_target"].isin([0, 1, 0.0, 1.0])
].copy()


# Clean text columns
df["claim_text"] = (
    df["claim_text"]
    .fillna("")
    .astype(str)
)

df["evidence_text"] = (
    df["evidence_text"]
    .fillna("")
    .astype(str)
)


# Create model input
df["model_text"] = (
    "Claim: "
    + df["claim_text"]
    + "\nEvidence: "
    + df["evidence_text"]
)


# Convert labels to integers
df["binary_target"] = df["binary_target"].astype(int)


# Keep only valid splits
valid_splits = ["train", "val", "test"]

df = df[df["split"].isin(valid_splits)].copy()


# Select output columns
output_columns = [
    "sample_id",
    "source_dataset",
    "split",
    "claim_text",
    "evidence_text",
    "model_text",
    "standardized_label",
    "binary_target",
]

# Keep only columns that exist
output_columns = [
    column for column in output_columns
    if column in df.columns
]

df = df[output_columns]


# Save each split
for split_name in valid_splits:
    split_df = df[df["split"] == split_name].copy()

    output_file = OUTPUT_DIR / f"claim_{split_name}.csv"

    split_df.to_csv(output_file, index=False)

    print(
        f"{split_name}: {len(split_df)} rows"
    )
    print(f"Saved to: {output_file}")


# Display label distribution
print("\nBinary label distribution:")
print(df["binary_target"].value_counts().sort_index())


# Final validation
print("\nMissing model text values:")
print(df["model_text"].isna().sum())

print("\nDataset preparation completed successfully.")