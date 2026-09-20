from pathlib import Path
import pandas as pd

BASE_DIR = Path(
    r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"
)

AUDIT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase5"
    / "phase5g_data_alignment_audit"
)

FILE_PATH = AUDIT_DIR / "duplicate_claims_across_splits.csv"

print("=" * 70)
print("PHASE 5G.2 - CROSS-SPLIT DUPLICATE CLAIM INSPECTION")
print("=" * 70)

if not FILE_PATH.exists():
    raise FileNotFoundError(FILE_PATH)

df = pd.read_csv(FILE_PATH)

print(f"\nFile: {FILE_PATH}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

if df.empty:
    print("\nNo cross-split duplicate claims found.")
else:
    print("\nAll cross-split duplicate records:\n")
    print(df.to_string(index=False))

    print("\nSaved inspection report...")

    output_path = AUDIT_DIR / "duplicate_claims_detailed_inspection.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)