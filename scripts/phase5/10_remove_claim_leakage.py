from pathlib import Path
import pandas as pd
import re
import json

# ============================================================
# PHASE 5H - CLAIM LEAKAGE REMOVAL
# ============================================================

BASE_DIR = Path(
    r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"
)

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "phase5"
    / "datasets"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase5"
    / "phase5h_leakage_cleaning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "train": DATA_DIR / "claim_train.csv",
    "validation": DATA_DIR / "claim_val.csv",
    "test": DATA_DIR / "claim_test.csv",
}

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "claim_train_clean.csv",
    "validation": OUTPUT_DIR / "claim_val_clean.csv",
    "test": OUTPUT_DIR / "claim_test_clean.csv",
}

# Existing split gets priority when duplicate claims occur.
# This preserves the original training data where possible.
SPLIT_PRIORITY = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


# ============================================================
# CLAIM NORMALIZATION
# ============================================================

def normalize_claim(text):
    """
    Normalize claims for duplicate detection.

    This does not alter the original claim_text column.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 5H - CLAIM LEAKAGE REMOVAL")
print("=" * 70)

dataframes = {}

for split_name, file_path in INPUT_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["loaded_split"] = split_name
    df["claim_clean"] = df["claim_text"].apply(normalize_claim)

    dataframes[split_name] = df

    print(
        f"{split_name.capitalize()} records: {len(df)}"
    )


combined_df = pd.concat(
    dataframes.values(),
    ignore_index=True
)

print(f"\nCombined records: {len(combined_df)}")


# ============================================================
# IDENTIFY CROSS-SPLIT DUPLICATE CLAIMS
# ============================================================

claim_split_counts = (
    combined_df
    .groupby("claim_clean")["loaded_split"]
    .nunique()
)

cross_split_claims = claim_split_counts[
    claim_split_counts > 1
].index

cross_split_df = combined_df[
    combined_df["claim_clean"].isin(cross_split_claims)
].copy()

print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATE ANALYSIS")
print("=" * 70)

print(
    f"Cross-split duplicate claim groups: "
    f"{len(cross_split_claims)}"
)

print(
    f"Rows involved: {len(cross_split_df)}"
)


# ============================================================
# SELECT RETAINED SPLIT
# ============================================================

# For every duplicated claim group, retain the split
# with the highest priority:
# train > validation > test

retained_split_map = {}

for claim, group in cross_split_df.groupby("claim_clean"):

    available_splits = group["loaded_split"].unique()

    retained_split = min(
        available_splits,
        key=lambda split: SPLIT_PRIORITY[split]
    )

    retained_split_map[claim] = retained_split


# ============================================================
# REMOVE CROSS-SPLIT DUPLICATE ROWS
# ============================================================

cleaned_df = combined_df.copy()

rows_to_remove = []

for index, row in cleaned_df.iterrows():

    claim = row["claim_clean"]
    current_split = row["loaded_split"]

    if claim in retained_split_map:

        retained_split = retained_split_map[claim]

        if current_split != retained_split:
            rows_to_remove.append(index)


cleaned_df = cleaned_df.drop(
    index=rows_to_remove
).reset_index(drop=True)

print(
    f"Rows removed: {len(rows_to_remove)}"
)

print(
    f"Records remaining: {len(cleaned_df)}"
)


# ============================================================
# SAVE CROSS-SPLIT REMOVAL REPORT
# ============================================================

if not cross_split_df.empty:

    cross_split_df["retained_split"] = (
        cross_split_df["claim_clean"]
        .map(retained_split_map)
    )

    cross_split_df["removed"] = (
        cross_split_df["loaded_split"]
        != cross_split_df["retained_split"]
    )

cross_split_report_path = (
    OUTPUT_DIR
    / "cross_split_leakage_removal_report.csv"
)

cross_split_df.to_csv(
    cross_split_report_path,
    index=False
)

print(
    f"\nLeakage report saved:\n"
    f"{cross_split_report_path}"
)


# ============================================================
# SAVE CLEANED SPLITS
# ============================================================

for split_name, output_path in OUTPUT_FILES.items():

    split_df = cleaned_df[
        cleaned_df["loaded_split"] == split_name
    ].copy()

    # Remove helper columns from final datasets
    split_df = split_df.drop(
        columns=["loaded_split", "claim_clean"],
        errors="ignore"
    )

    split_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"{split_name.capitalize()} cleaned records: "
        f"{len(split_df)}"
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# FINAL LEAKAGE VERIFICATION
# ============================================================

verification_df = cleaned_df.copy()

verification = (
    verification_df
    .groupby("claim_clean")["loaded_split"]
    .nunique()
)

remaining_leakage = verification[
    verification > 1
]

print("\n" + "=" * 70)
print("FINAL LEAKAGE VERIFICATION")
print("=" * 70)

print(
    f"Remaining cross-split duplicate groups: "
    f"{len(remaining_leakage)}"
)

if len(remaining_leakage) == 0:
    print("PASS: No claim appears across multiple splits.")
else:
    print(
        "WARNING: Cross-split duplicate claims remain."
    )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "original_total_records": int(len(combined_df)),
    "cross_split_duplicate_groups": int(len(cross_split_claims)),
    "rows_removed": int(len(rows_to_remove)),
    "cleaned_total_records": int(len(cleaned_df)),
    "remaining_cross_split_groups": int(len(remaining_leakage)),
    "split_counts": {
        split: int(
            (cleaned_df["loaded_split"] == split).sum()
        )
        for split in ["train", "validation", "test"]
    },
}

summary_path = OUTPUT_DIR / "phase5h_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print(
    f"\nSummary saved:\n{summary_path}"
)

print("\n" + "=" * 70)
print("PHASE 5H COMPLETED")
print("=" * 70)