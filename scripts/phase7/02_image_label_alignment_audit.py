
from pathlib import Path
import pandas as pd
import json

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase7"
    / "phase7b_image_label_alignment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 7B: IMAGE-LABEL ALIGNMENT AUDIT")
print("=" * 70)

# Find all CSV files
csv_files = list(DATA_DIR.rglob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

results = []

for csv_path in csv_files:

    try:
        # Read only first 5 rows for schema inspection
        df = pd.read_csv(csv_path, nrows=5)

        columns = df.columns.tolist()

        # Detect image-related columns
        image_columns = [
            column
            for column in columns
            if any(
                term in column.lower()
                for term in [
                    "image",
                    "img",
                    "filename",
                    "file",
                    "path"
                ]
            )
        ]

        # Detect label-related columns
        label_columns = [
            column
            for column in columns
            if any(
                term in column.lower()
                for term in [
                    "label",
                    "target",
                    "class",
                    "category",
                    "truth"
                ]
            )
        ]

        # Detect split-related columns
        split_columns = [
            column
            for column in columns
            if "split" in column.lower()
        ]

        # Save audit result
        results.append({
            "csv_path": str(csv_path.relative_to(BASE_DIR)),
            "columns": " | ".join(columns),
            "image_related_columns": " | ".join(image_columns),
            "label_related_columns": " | ".join(label_columns),
            "split_columns": " | ".join(split_columns),
            "rows_sampled": len(df)
        })

        # Display results
        print("\n" + "-" * 70)
        print(f"File: {csv_path.relative_to(BASE_DIR)}")
        print(f"Columns: {columns}")

        if image_columns:
            print(f"Image-related columns: {image_columns}")

        if label_columns:
            print(f"Label-related columns: {label_columns}")

        if split_columns:
            print(f"Split columns: {split_columns}")

    except Exception as error:

        print(f"\nCould not read: {csv_path}")
        print(f"Error: {error}")


# Create audit DataFrame
audit_df = pd.DataFrame(results)

# Save CSV audit
audit_path = OUTPUT_DIR / "csv_schema_audit.csv"

audit_df.to_csv(
    audit_path,
    index=False
)

# Create summary
summary = {
    "csv_files_found": len(csv_files),

    "files_with_image_columns": int(
        (
            audit_df["image_related_columns"].str.len() > 0
        ).sum()
    ) if not audit_df.empty else 0,

    "files_with_label_columns": int(
        (
            audit_df["label_related_columns"].str.len() > 0
        ).sum()
    ) if not audit_df.empty else 0,

    "files_with_split_columns": int(
        (
            audit_df["split_columns"].str.len() > 0
        ).sum()
    ) if not audit_df.empty else 0
}

# Save JSON summary
summary_path = OUTPUT_DIR / "phase7b_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)

print(f"\nAudit saved to: {audit_path}")
print(f"Summary saved to: {summary_path}")

print("\nPHASE 7B COMPLETED SUCCESSFULLY")
