import os
import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "phase1"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# 1. PUBHEALTH
# ============================================================

def explore_pubhealth():

    print("\n" + "=" * 60)
    print("PUBHEALTH DATASET")
    print("=" * 60)

    pubhealth_dir = RAW_DIR / "pubhealth"

    train_path = pubhealth_dir / "train.tsv"
    dev_path = pubhealth_dir / "dev.tsv"
    test_path = pubhealth_dir / "test.tsv"

    train = pd.read_csv(train_path, sep="\t")
    dev = pd.read_csv(dev_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")

    print("\nTrain shape:", train.shape)
    print("Dev shape:", dev.shape)
    print("Test shape:", test.shape)

    print("\nColumns:")
    print(train.columns.tolist())

    print("\nLabel distribution - Train:")
    print(train["label"].value_counts(dropna=False))

    print("\nMissing values - Train:")
    print(train.isnull().sum())

    print("\nDuplicate rows - Train:")
    print(train.duplicated().sum())

    # Save label counts
    label_counts = (
        train["label"]
        .value_counts(dropna=False)
        .reset_index()
    )

    label_counts.columns = ["label", "count"]

    save_csv(
        label_counts,
        "pubhealth_label_counts.csv"
    )

    # Create manifest
    manifest = pd.DataFrame()

    manifest["sample_id"] = train["claim_id"].astype(str)
    manifest["dataset"] = "PubHealth"
    manifest["claim"] = train["claim"].fillna("")
    manifest["explanation"] = train["explanation"].fillna("")
    manifest["main_text"] = train["main_text"].fillna("")
    manifest["sources"] = train["sources"].fillna("")
    manifest["original_label"] = train["label"].fillna("")
    manifest["subjects"] = train["subjects"].fillna("")
    manifest["split"] = "train"

    dev_manifest = pd.DataFrame()
    dev_manifest["sample_id"] = dev["claim_id"].astype(str)
    dev_manifest["dataset"] = "PubHealth"
    dev_manifest["claim"] = dev["claim"].fillna("")
    dev_manifest["explanation"] = dev["explanation"].fillna("")
    dev_manifest["main_text"] = dev["main_text"].fillna("")
    dev_manifest["sources"] = dev["sources"].fillna("")
    dev_manifest["original_label"] = dev["label"].fillna("")
    dev_manifest["subjects"] = dev["subjects"].fillna("")
    dev_manifest["split"] = "dev"

    test_manifest = pd.DataFrame()
    test_manifest["sample_id"] = test["claim_id"].astype(str)
    test_manifest["dataset"] = "PubHealth"
    test_manifest["claim"] = test["claim"].fillna("")
    test_manifest["explanation"] = test["explanation"].fillna("")
    test_manifest["main_text"] = test["main_text"].fillna("")
    test_manifest["sources"] = test["sources"].fillna("")
    test_manifest["original_label"] = test["label"].fillna("")
    test_manifest["subjects"] = test["subjects"].fillna("")
    test_manifest["split"] = "test"

    full_manifest = pd.concat(
        [manifest, dev_manifest, test_manifest],
        ignore_index=True
    )

    path = MANIFEST_DIR / "pubhealth_manifest.csv"
    full_manifest.to_csv(path, index=False)

    print(f"\nSaved PubHealth manifest: {path}")
    print("Manifest shape:", full_manifest.shape)


# ============================================================
# 2. HEALTHFC
# ============================================================

def explore_healthfc():

    print("\n" + "=" * 60)
    print("HEALTHFC DATASET")
    print("=" * 60)

    path = RAW_DIR / "healthfc" / "Datensatz.csv"

    df = pd.read_csv(path)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Save label counts
    label_counts = (
        df["label"]
        .value_counts(dropna=False)
        .reset_index()
    )

    label_counts.columns = ["label", "count"]

    save_csv(
        label_counts,
        "healthfc_label_counts.csv"
    )

    # Create manifest
    manifest = pd.DataFrame()

    manifest["sample_id"] = range(len(df))
    manifest["dataset"] = "HealthFC"

    manifest["claim"] = df["en_claim"].fillna("")
    manifest["explanation"] = df["en_explanation"].fillna("")
    manifest["top_sentences"] = df["en_top_sentences"].fillna("")

    manifest["original_label"] = df["label"]
    manifest["verdict"] = df["de_verdict"].fillna("")

    manifest["title"] = df["de_title"].fillna("")
    manifest["authors"] = df["authors"].fillna("")
    manifest["date"] = df["date"].fillna("")
    manifest["url"] = df["url"].fillna("")

    path = MANIFEST_DIR / "healthfc_manifest.csv"
    manifest.to_csv(path, index=False)

    print(f"\nSaved HealthFC manifest: {path}")
    print("Manifest shape:", manifest.shape)


# ============================================================
# 3. CHARTQA
# ============================================================

def explore_chartqa():

    print("\n" + "=" * 60)
    print("CHARTQA DATASET")
    print("=" * 60)

    chartqa_dir = RAW_DIR / "chartqa" / "ChartQA Dataset"

    if not chartqa_dir.exists():
        print("ChartQA directory not found:")
        print(chartqa_dir)
        return

    # List files
    all_files = [
        p for p in chartqa_dir.rglob("*")
        if p.is_file()
    ]

    print("\nTotal files:", len(all_files))

    # Count extensions
    extension_counts = {}

    for file in all_files:
        ext = file.suffix.lower()

        if ext == "":
            ext = "[no extension]"

        extension_counts[ext] = (
            extension_counts.get(ext, 0) + 1
        )

    print("\nFile types:")

    for ext, count in sorted(
        extension_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{ext}: {count}")

    # Count files by split
    split_data = []

    for split in ["train", "val", "test"]:

        split_dir = chartqa_dir / split

        if not split_dir.exists():
            continue

        files = [
            p for p in split_dir.rglob("*")
            if p.is_file()
        ]

        split_data.append({
            "dataset": "ChartQA",
            "split": split,
            "file_count": len(files)
        })

    split_df = pd.DataFrame(split_data)

    if not split_df.empty:
        print("\nFiles by split:")
        print(split_df.to_string(index=False))

        save_csv(
            split_df,
            "chartqa_summary.csv"
        )

    # Create basic manifest from image/table files
    manifest_rows = []

    for split in ["train", "val", "test"]:

        split_dir = chartqa_dir / split

        if not split_dir.exists():
            continue

        for file in split_dir.rglob("*"):

            if not file.is_file():
                continue

            relative_path = file.relative_to(PROJECT_ROOT)

            manifest_rows.append({
                "sample_id": file.stem,
                "dataset": "ChartQA",
                "split": split,
                "file_path": str(relative_path),
                "file_type": file.suffix.lower()
            })

    manifest = pd.DataFrame(manifest_rows)

    path = MANIFEST_DIR / "chartqa_manifest.csv"
    manifest.to_csv(path, index=False)

    print(f"\nSaved ChartQA manifest: {path}")
    print("Manifest shape:", manifest.shape)


# ============================================================
# 4. OVERALL DATASET SUMMARY
# ============================================================

def create_summary():

    summary = []

    # PubHealth
    pubhealth_train = pd.read_csv(
        RAW_DIR / "pubhealth" / "train.tsv",
        sep="\t"
    )

    pubhealth_dev = pd.read_csv(
        RAW_DIR / "pubhealth" / "dev.tsv",
        sep="\t"
    )

    pubhealth_test = pd.read_csv(
        RAW_DIR / "pubhealth" / "test.tsv",
        sep="\t"
    )

    summary.append({
        "dataset": "PubHealth",
        "train_samples": len(pubhealth_train),
        "validation_samples": len(pubhealth_dev),
        "test_samples": len(pubhealth_test),
        "total_samples":
            len(pubhealth_train)
            + len(pubhealth_dev)
            + len(pubhealth_test)
    })

    # HealthFC
    healthfc = pd.read_csv(
        RAW_DIR / "healthfc" / "Datensatz.csv"
    )

    summary.append({
        "dataset": "HealthFC",
        "train_samples": "",
        "validation_samples": "",
        "test_samples": "",
        "total_samples": len(healthfc)
    })

    # ChartQA
    chartqa_dir = RAW_DIR / "chartqa" / "ChartQA Dataset"

    chartqa_counts = {}

    for split in ["train", "val", "test"]:

        split_dir = chartqa_dir / split

        if split_dir.exists():

            chartqa_counts[split] = len([
                p for p in split_dir.rglob("*")
                if p.is_file()
            ])

        else:
            chartqa_counts[split] = 0

    summary.append({
        "dataset": "ChartQA",
        "train_samples": chartqa_counts.get("train", 0),
        "validation_samples": chartqa_counts.get("val", 0),
        "test_samples": chartqa_counts.get("test", 0),
        "total_samples":
            sum(chartqa_counts.values())
    })

    summary_df = pd.DataFrame(summary)

    save_csv(
        summary_df,
        "dataset_summary.csv"
    )

    print("\n" + "=" * 60)
    print("OVERALL DATASET SUMMARY")
    print("=" * 60)

    print(summary_df.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("PHASE 1 - DATASET EXPLORATION")
    print("=" * 60)

    explore_pubhealth()
    explore_healthfc()
    explore_chartqa()
    create_summary()

    print("\n")
    print("=" * 60)
    print("PHASE 1 EXPLORATION COMPLETED")
    print("=" * 60)

    print("\nCheck these folders:")
    print("data/interim/phase1/")
    print("data/manifests/")