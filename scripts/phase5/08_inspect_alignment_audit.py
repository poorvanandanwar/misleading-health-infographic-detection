from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 5G.1 - DATA ALIGNMENT AUDIT INSPECTION
# ============================================================

BASE_DIR = Path(
    r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"
)

AUDIT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase5"
    / "phase5g_data_alignment_audit"
)

FILES_TO_INSPECT = {
    "duplicate_claims_across_splits": "duplicate_claims_across_splits.csv",
    "non_health_candidates": "non_health_candidate_rows.csv",
    "reused_evidence": "reused_evidence_analysis.csv",
    "low_overlap": "potential_mismatches_low_overlap.csv",
    "overlap_audit": "claim_evidence_overlap_audit.csv",
    "missing_values": "missing_values_report.csv",
}

print("=" * 70)
print("PHASE 5G.1 - DATA ALIGNMENT AUDIT INSPECTION")
print("=" * 70)

print(f"\nAudit directory:\n{AUDIT_DIR}")

if not AUDIT_DIR.exists():
    raise FileNotFoundError(
        f"Audit directory not found: {AUDIT_DIR}"
    )

for report_name, filename in FILES_TO_INSPECT.items():

    file_path = AUDIT_DIR / filename

    print("\n" + "=" * 70)
    print(f"REPORT: {report_name.upper()}")
    print(f"FILE: {filename}")
    print("=" * 70)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)

        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        if df.empty:
            print("The file is empty.")
            continue

        print("\nFirst 5 records:")
        print(df.head(5).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

    except Exception as error:
        print(f"Error reading {filename}: {error}")


# ============================================================
# DETAILED INSPECTION OF CROSS-SPLIT DUPLICATES
# ============================================================

duplicate_file = AUDIT_DIR / "duplicate_claims_across_splits.csv"

if duplicate_file.exists():

    print("\n" + "=" * 70)
    print("CROSS-SPLIT DUPLICATE CLAIM INSPECTION")
    print("=" * 70)

    duplicates = pd.read_csv(duplicate_file)

    print(f"Number of rows: {len(duplicates)}")

    if not duplicates.empty:
        print(duplicates.to_string(index=False))
    else:
        print("No cross-split duplicate claims found.")


# ============================================================
# DETAILED INSPECTION OF NON-HEALTH CANDIDATES
# ============================================================

non_health_file = AUDIT_DIR / "non_health_candidate_rows.csv"

if non_health_file.exists():

    print("\n" + "=" * 70)
    print("NON-HEALTH CANDIDATE INSPECTION")
    print("=" * 70)

    non_health = pd.read_csv(non_health_file)

    print(f"Total candidates: {len(non_health)}")

    if not non_health.empty:

        print("\nSource dataset distribution:")
        if "source_dataset" in non_health.columns:
            print(
                non_health["source_dataset"]
                .value_counts(dropna=False)
                .to_string()
            )

        print("\nLabel distribution:")
        if "standardized_label" in non_health.columns:
            print(
                non_health["standardized_label"]
                .value_counts(dropna=False)
                .to_string()
            )

        print("\nFirst 20 candidate claims:")
        display_columns = [
            column
            for column in [
                "sample_id",
                "source_dataset",
                "claim_text",
                "evidence_text",
                "standardized_label",
                "binary_target",
            ]
            if column in non_health.columns
        ]

        print(
            non_health[display_columns]
            .head(20)
            .to_string(index=False)
        )


# ============================================================
# DETAILED INSPECTION OF REUSED EVIDENCE
# ============================================================

reused_file = AUDIT_DIR / "reused_evidence_analysis.csv"

if reused_file.exists():

    print("\n" + "=" * 70)
    print("REUSED EVIDENCE INSPECTION")
    print("=" * 70)

    reused = pd.read_csv(reused_file)

    print(f"Total rows: {len(reused)}")

    if not reused.empty:
        print(reused.head(20).to_string(index=False))


print("\n" + "=" * 70)
print("PHASE 5G.1 INSPECTION COMPLETED")
print("=" * 70)