
from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6b_source_wise_error_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6c_qualitative_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_PATH = INPUT_DIR / "source_wise_test_errors.csv"


# ============================================================
# 2. LOAD ERROR DATA
# ============================================================

print("=" * 70)
print("PHASE 6C - QUALITATIVE ERROR ANALYSIS")
print("=" * 70)

errors_df = pd.read_csv(ERRORS_PATH)

print(f"\nTotal incorrect predictions: {len(errors_df)}")

print("\nErrors by source:")
print(errors_df["source_dataset"].value_counts())

print("\nErrors by type:")
print(errors_df["error_type"].value_counts())


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


errors_df["claim_text"] = errors_df["claim_text"].apply(clean_text)
errors_df["evidence_text"] = errors_df["evidence_text"].apply(clean_text)

errors_df["combined_text"] = (
    errors_df["claim_text"] + " " +
    errors_df["evidence_text"]
)


# ============================================================
# 4. TEXT FEATURES
# ============================================================

errors_df["claim_word_count"] = (
    errors_df["claim_text"]
    .str.split()
    .str.len()
)

errors_df["evidence_word_count"] = (
    errors_df["evidence_text"]
    .str.split()
    .str.len()
)

errors_df["combined_word_count"] = (
    errors_df["combined_text"]
    .str.split()
    .str.len()
)


# ============================================================
# 5. LINGUISTIC INDICATORS
# ============================================================

indicator_patterns = {
    "uncertainty_language": [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bpotentially\b",
        r"\bperhaps\b",
        r"\buncertain\b"
    ],

    "strong_claim_language": [
        r"\bproves?\b",
        r"\bguarantee[sd]?\b",
        r"\bdefinitely\b",
        r"\balways\b",
        r"\bnever\b",
        r"\bcompletely\b",
        r"\bmiracle\b"
    ],

    "medical_treatment_language": [
        r"\btreatment\b",
        r"\bdrug\b",
        r"\bmedicine\b",
        r"\btherapy\b",
        r"\bcure\b",
        r"\bvaccine\b",
        r"\bsupplement\b"
    ],

    "numerical_information": [
        r"\b\d+(\.\d+)?%\b",
        r"\b\d+(\.\d+)?\b",
        r"\bpercent\b",
        r"\bmillion\b",
        r"\bbillion\b"
    ],

    "comparative_language": [
        r"\bmore\b",
        r"\bless\b",
        r"\bbetter\b",
        r"\bworse\b",
        r"\bhigher\b",
        r"\blower\b",
        r"\bincrease[sd]?\b",
        r"\bdecrease[sd]?\b"
    ],

    "causal_language": [
        r"\bcauses?\b",
        r"\bleads? to\b",
        r"\bdue to\b",
        r"\bresults? in\b",
        r"\bprevents?\b",
        r"\bprotects?\b"
    ]
}


def contains_indicator(text, patterns):
    text = text.lower()

    for pattern in patterns:
        if re.search(pattern, text):
            return 1

    return 0


for indicator_name, patterns in indicator_patterns.items():

    errors_df[indicator_name] = errors_df[
        "combined_text"
    ].apply(
        lambda text: contains_indicator(text, patterns)
    )


# ============================================================
# 6. ERROR PATTERN ASSIGNMENT
# ============================================================

def assign_error_pattern(row):

    claim = row["claim_text"].lower()
    evidence = row["evidence_text"].lower()

    if row["uncertainty_language"] == 1:
        return "Uncertainty or hedging language"

    elif row["strong_claim_language"] == 1:
        return "Strong or exaggerated claim language"

    elif row["causal_language"] == 1:
        return "Causal relationship language"

    elif row["numerical_information"] == 1:
        return "Numerical or statistical claim"

    elif row["comparative_language"] == 1:
        return "Comparative claim"

    elif row["medical_treatment_language"] == 1:
        return "Medical treatment terminology"

    elif len(claim.split()) < 8:
        return "Short or context-limited claim"

    elif len(evidence.split()) < 20:
        return "Limited evidence context"

    else:
        return "Other or unclear pattern"


errors_df["error_pattern"] = errors_df.apply(
    assign_error_pattern,
    axis=1
)


# ============================================================
# 7. PATTERN SUMMARY
# ============================================================

pattern_summary = (
    errors_df
    .groupby(
        ["source_dataset", "error_type", "error_pattern"],
        observed=False
    )
    .agg(
        count=("sample_id", "count"),
        average_claim_words=("claim_word_count", "mean"),
        average_evidence_words=("evidence_word_count", "mean"),
        average_probability=("probability_positive", "mean")
    )
    .reset_index()
    .sort_values(
        by=["source_dataset", "count"],
        ascending=[True, False]
    )
)

pattern_summary.to_csv(
    OUTPUT_DIR / "qualitative_error_pattern_summary.csv",
    index=False
)


# ============================================================
# 8. ERROR TYPE + TEXT STATISTICS
# ============================================================

text_statistics = (
    errors_df
    .groupby(
        ["source_dataset", "error_type"],
        observed=False
    )
    .agg(
        number_of_errors=("sample_id", "count"),
        average_claim_length=("claim_word_count", "mean"),
        average_evidence_length=("evidence_word_count", "mean"),
        minimum_claim_length=("claim_word_count", "min"),
        maximum_claim_length=("claim_word_count", "max"),
        average_probability=("probability_positive", "mean")
    )
    .reset_index()
)

text_statistics.to_csv(
    OUTPUT_DIR / "error_text_statistics.csv",
    index=False
)


# ============================================================
# 9. REPRESENTATIVE ERROR EXAMPLES
# ============================================================

representative_columns = [
    "sample_id",
    "source_dataset",
    "claim_text",
    "evidence_text",
    "standardized_label",
    "binary_target",
    "predicted_label",
    "probability_positive",
    "confidence_group",
    "error_type",
    "error_pattern",
    "claim_word_count",
    "evidence_word_count"
]

representative_errors = (
    errors_df[representative_columns]
    .sort_values(
        by=["source_dataset", "probability_positive"]
    )
)

representative_errors.to_csv(
    OUTPUT_DIR / "representative_error_examples.csv",
    index=False
)


# ============================================================
# 10. HIGH-CONFIDENCE ERRORS
# ============================================================

high_confidence_errors = errors_df[
    (
        (
            errors_df["error_type"] == "False Positive"
        )
        & (
            errors_df["probability_positive"] >= 0.80
        )
    )
    |
    (
        (
            errors_df["error_type"] == "False Negative"
        )
        & (
            errors_df["probability_positive"] <= 0.20
        )
    )
].copy()

high_confidence_errors[
    representative_columns
].to_csv(
    OUTPUT_DIR / "high_confidence_errors.csv",
    index=False
)


# ============================================================
# 11. ERROR PATTERN COUNTS
# ============================================================

pattern_counts = (
    errors_df["error_pattern"]
    .value_counts()
    .reset_index()
)

pattern_counts.columns = [
    "error_pattern",
    "count"
]

pattern_counts.to_csv(
    OUTPUT_DIR / "overall_error_pattern_counts.csv",
    index=False
)


# ============================================================
# 12. SUMMARY JSON
# ============================================================

summary = {
    "phase": "6C",
    "title": "Qualitative Error Analysis",
    "total_errors": int(len(errors_df)),
    "false_positives": int(
        (
            errors_df["error_type"] == "False Positive"
        ).sum()
    ),
    "false_negatives": int(
        (
            errors_df["error_type"] == "False Negative"
        ).sum()
    ),
    "source_distribution": (
        errors_df["source_dataset"]
        .value_counts()
        .to_dict()
    ),
    "error_pattern_distribution": (
        errors_df["error_pattern"]
        .value_counts()
        .to_dict()
    ),
    "high_confidence_errors": int(
        len(high_confidence_errors)
    )
}

with open(
    OUTPUT_DIR / "phase6c_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4,
        default=int
    )


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("QUALITATIVE ERROR PATTERNS")
print("=" * 70)

print(
    pattern_counts.to_string(index=False)
)

print("\n" + "=" * 70)
print("ERROR TEXT STATISTICS")
print("=" * 70)

print(
    text_statistics.to_string(index=False)
)

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE ERRORS")
print("=" * 70)

print(
    f"High-confidence errors: {len(high_confidence_errors)}"
)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR}")
print("- qualitative_error_pattern_summary.csv")
print("- error_text_statistics.csv")
print("- representative_error_examples.csv")
print("- high_confidence_errors.csv")
print("- overall_error_pattern_counts.csv")
print("- phase6c_summary.json")

print("\nPhase 6C completed successfully.")