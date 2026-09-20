import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import Counter

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"

from pathlib import Path

PROJECT_ROOT = Path(BASE_DIR)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase5" / "datasets"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase5"
    / "phase5g_data_alignment_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "claim_train.csv",
    "validation": DATA_DIR / "claim_val.csv",
    "test": DATA_DIR / "claim_test.csv",
}
# ============================================================
# SETTINGS
# ============================================================

REQUIRED_COLUMNS = [
    "sample_id",
    "source_dataset",
    "split",
    "claim_text",
    "evidence_text",
    "model_text",
    "standardized_label",
    "binary_target",
]

HEALTH_KEYWORDS = [
    "health", "disease", "medical", "medicine", "doctor",
    "patient", "symptom", "treatment", "cancer", "virus",
    "vaccine", "infection", "drug", "therapy", "blood",
    "heart", "lung", "brain", "disease", "illness",
    "nutrition", "diet", "obesity", "diabetes", "pregnancy",
    "mental", "clinical", "hospital", "death", "cure",
    "prevent", "risk", "study", "research", "healthcare"
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "is", "are", "was", "were", "be", "been", "to",
    "of", "in", "on", "for", "with", "by", "as", "at",
    "from", "that", "this", "it", "its", "they", "their",
    "can", "may", "might", "could", "should", "has", "have"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def tokenize(text):
    words = clean_text(text).split()
    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    }


def calculate_jaccard(claim, evidence):
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)

    if not claim_tokens or not evidence_tokens:
        return 0.0

    intersection = claim_tokens.intersection(evidence_tokens)
    union = claim_tokens.union(evidence_tokens)

    return len(intersection) / len(union)


def health_keyword_present(text):
    text = clean_text(text)
    return int(any(keyword in text for keyword in HEALTH_KEYWORDS))


# ============================================================
# LOAD DATA
# ============================================================

datasets = []

for split_name, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: {missing_columns}"
        )

    df["loaded_split"] = split_name
    datasets.append(df)

combined = pd.concat(datasets, ignore_index=True)

print("=" * 70)
print("PHASE 5G - DATA ALIGNMENT AUDIT")
print("=" * 70)

print(f"Total records: {len(combined)}")
print(f"Train records: {len(datasets[0])}")
print(f"Validation records: {len(datasets[1])}")
print(f"Test records: {len(datasets[2])}")

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

overview = (
    combined
    .groupby(["loaded_split", "source_dataset", "binary_target"])
    .size()
    .reset_index(name="record_count")
)

overview.to_csv(
    OUTPUT_DIR / "dataset_overview.csv",
    index=False
)

# ============================================================
# 2. MISSING VALUES AND EMPTY TEXT
# ============================================================

missing_records = []

for column in REQUIRED_COLUMNS:

    missing_count = int(combined[column].isna().sum())

    empty_count = 0

    if combined[column].dtype == "object":
        empty_count = int(
            combined[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    missing_records.append({
        "column": column,
        "missing_count": missing_count,
        "empty_count": empty_count
    })

missing_report = pd.DataFrame(missing_records)

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index=False
)

# ============================================================
# 3. DUPLICATE SAMPLE IDS
# ============================================================

duplicate_ids = combined[
    combined["sample_id"].duplicated(keep=False)
].sort_values("sample_id")

duplicate_ids.to_csv(
    OUTPUT_DIR / "duplicate_sample_ids.csv",
    index=False
)

# ============================================================
# 4. DUPLICATE CLAIMS ACROSS SPLITS
# ============================================================

combined["claim_clean"] = (
    combined["claim_text"]
    .fillna("")
    .astype(str)
    .map(clean_text)
)

duplicate_claims = combined[
    combined["claim_clean"].duplicated(keep=False)
].sort_values("claim_clean")

duplicate_claims_across_splits = (
    duplicate_claims
    .groupby("claim_clean")["loaded_split"]
    .nunique()
)

cross_split_claims = duplicate_claims[
    duplicate_claims["claim_clean"].isin(
        duplicate_claims_across_splits[
            duplicate_claims_across_splits > 1
        ].index
    )
]

cross_split_claims.to_csv(
    OUTPUT_DIR / "duplicate_claims_across_splits.csv",
    index=False
)

# ============================================================
# 5. CLAIM-EVIDENCE OVERLAP AUDIT
# ============================================================

audit = combined[
    [
        "sample_id",
        "source_dataset",
        "loaded_split",
        "claim_text",
        "evidence_text",
        "standardized_label",
        "binary_target"
    ]
].copy()

audit["claim_token_count"] = audit["claim_text"].fillna("").map(
    lambda x: len(tokenize(x))
)

audit["evidence_token_count"] = audit["evidence_text"].fillna("").map(
    lambda x: len(tokenize(x))
)

audit["jaccard_overlap"] = audit.apply(
    lambda row: calculate_jaccard(
        row["claim_text"],
        row["evidence_text"]
    ),
    axis=1
)

audit["claim_has_health_keyword"] = (
    audit["claim_text"]
    .fillna("")
    .map(health_keyword_present)
)

audit["evidence_has_health_keyword"] = (
    audit["evidence_text"]
    .fillna("")
    .map(health_keyword_present)
)

audit["low_overlap_candidate"] = (
    (audit["claim_token_count"] > 0)
    & (audit["evidence_token_count"] > 0)
    & (audit["jaccard_overlap"] < 0.05)
)

audit.to_csv(
    OUTPUT_DIR / "claim_evidence_overlap_audit.csv",
    index=False
)

# ============================================================
# 6. POTENTIAL CLAIM-EVIDENCE MISMATCHES
# ============================================================

low_overlap = audit[
    audit["low_overlap_candidate"]
].copy()

low_overlap = low_overlap.sort_values(
    "jaccard_overlap",
    ascending=True
)

low_overlap.to_csv(
    OUTPUT_DIR / "potential_mismatches_low_overlap.csv",
    index=False
)

# ============================================================
# 7. POSSIBLE NON-HEALTH CLAIMS
# ============================================================

non_health_candidates = audit[
    audit["claim_has_health_keyword"] == 0
].copy()

non_health_candidates.to_csv(
    OUTPUT_DIR / "non_health_candidate_rows.csv",
    index=False
)

# ============================================================
# 8. EVIDENCE REUSE ANALYSIS
# ============================================================

combined["evidence_clean"] = (
    combined["evidence_text"]
    .fillna("")
    .astype(str)
    .map(clean_text)
)

evidence_usage = (
    combined
    .groupby("evidence_clean")
    .agg(
        evidence_usage_count=("sample_id", "count"),
        unique_claim_count=("claim_clean", "nunique"),
        unique_split_count=("loaded_split", "nunique")
    )
    .reset_index()
)

reused_evidence = evidence_usage[
    evidence_usage["evidence_usage_count"] > 1
].sort_values(
    "evidence_usage_count",
    ascending=False
)

reused_evidence.to_csv(
    OUTPUT_DIR / "reused_evidence_analysis.csv",
    index=False
)

# ============================================================
# 9. SUMMARY JSON
# ============================================================

summary = {
    "total_records": int(len(combined)),
    "duplicate_sample_id_rows": int(len(duplicate_ids)),
    "duplicate_claim_rows": int(len(duplicate_claims)),
    "cross_split_duplicate_claim_rows": int(len(cross_split_claims)),
    "low_overlap_candidates": int(len(low_overlap)),
    "non_health_candidates": int(len(non_health_candidates)),
    "reused_evidence_entries": int(len(reused_evidence)),
    "missing_values": {
        row["column"]: {
            "missing_count": int(row["missing_count"]),
            "empty_count": int(row["empty_count"])
        }
        for _, row in missing_report.iterrows()
    }
}

with open(
    OUTPUT_DIR / "phase5g_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\nDATA ALIGNMENT RESULTS")
print("-" * 70)

print(f"Duplicate sample ID rows: {len(duplicate_ids)}")
print(f"Duplicate claim rows: {len(duplicate_claims)}")
print(
    f"Cross-split duplicate claim rows: "
    f"{len(cross_split_claims)}"
)
print(f"Low-overlap candidates: {len(low_overlap)}")
print(f"Non-health candidates: {len(non_health_candidates)}")
print(f"Reused evidence entries: {len(reused_evidence)}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print(
    "Low lexical overlap is only a review candidate. "
    "It does not automatically prove that evidence is mismatched."
)

print("\nPhase 5G audit completed successfully.")