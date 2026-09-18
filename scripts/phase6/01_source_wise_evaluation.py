
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\aditi\Desktop\DL PROJECT\misleading-health-infographic-detection"
)

DATA_DIR = (
    BASE_DIR
    / "outputs"
    / "phase5"
    / "phase5h_leakage_cleaning"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase6"
    / "phase6a_source_wise_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

train_path = DATA_DIR / "claim_train_clean.csv"
val_path = DATA_DIR / "claim_val_clean.csv"
test_path = DATA_DIR / "claim_test_clean.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print("=" * 70)
print("PHASE 6A - SOURCE-WISE MODEL EVALUATION")
print("=" * 70)

print(f"\nTrain records: {len(train_df)}")
print(f"Validation records: {len(val_df)}")
print(f"Test records: {len(test_df)}")


# ============================================================
# TEXT PREPARATION
# ============================================================

def create_model_text(df):
    claim = df["claim_text"].fillna("").astype(str)
    evidence = df["evidence_text"].fillna("").astype(str)

    return "claim: " + claim + " evidence: " + evidence


X_train_text = create_model_text(train_df)
X_val_text = create_model_text(val_df)
X_test_text = create_model_text(test_df)

y_train = train_df["binary_target"].astype(int)
y_val = val_df["binary_target"].astype(int)
y_test = test_df["binary_target"].astype(int)


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VECTORIZATION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

print(f"\nTraining matrix: {X_train.shape}")
print(f"Validation matrix: {X_val.shape}")
print(f"Testing matrix: {X_test.shape}")


# ============================================================
# MODEL TRAINING
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("\nLogistic Regression training completed.")


# ============================================================
# SOURCE-WISE EVALUATION FUNCTION
# ============================================================

def evaluate_by_source(df, X, y, dataset_name):

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    results = []
    confusion_results = []

    sources = sorted(df["source_dataset"].dropna().unique())

    for source in sources:

        mask = df["source_dataset"].values == source

        source_y = y.values[mask]
        source_predictions = predictions[mask]
        source_probabilities = probabilities[mask]

        if len(np.unique(source_y)) < 2:
            source_auc = None
        else:
            source_auc = roc_auc_score(
                source_y,
                source_probabilities
            )

        tn, fp, fn, tp = confusion_matrix(
            source_y,
            source_predictions,
            labels=[0, 1]
        ).ravel()

        results.append({
            "dataset_split": dataset_name,
            "source_dataset": source,
            "samples": len(source_y),
            "accuracy": accuracy_score(
                source_y,
                source_predictions
            ),
            "precision": precision_score(
                source_y,
                source_predictions,
                zero_division=0
            ),
            "recall": recall_score(
                source_y,
                source_predictions,
                zero_division=0
            ),
            "f1_score": f1_score(
                source_y,
                source_predictions,
                zero_division=0
            ),
            "roc_auc": source_auc
        })

        confusion_results.append({
            "dataset_split": dataset_name,
            "source_dataset": source,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        })

    return results, confusion_results


# ============================================================
# EVALUATE TRAIN, VALIDATION, AND TEST
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-WISE EVALUATION")
print("=" * 70)

all_metrics = []
all_confusion_matrices = []

for df, X, y, split_name in [
    (train_df, X_train, y_train, "train"),
    (val_df, X_val, y_val, "validation"),
    (test_df, X_test, y_test, "test")
]:

    metrics, confusion_results = evaluate_by_source(
        df,
        X,
        y,
        split_name
    )

    all_metrics.extend(metrics)
    all_confusion_matrices.extend(confusion_results)


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
confusion_df = pd.DataFrame(all_confusion_matrices)

metrics_path = OUTPUT_DIR / "source_wise_metrics.csv"
confusion_path = OUTPUT_DIR / "source_wise_confusion_matrices.csv"

metrics_df.to_csv(metrics_path, index=False)
confusion_df.to_csv(confusion_path, index=False)


# ============================================================
# DISPLAY TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST SOURCE-WISE RESULTS")
print("=" * 70)

test_results = metrics_df[
    metrics_df["dataset_split"] == "test"
]

print(
    test_results.to_string(index=False)
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "phase": "6A",
    "description": "Source-wise evaluation of cleaned TF-IDF model",
    "train_records": len(train_df),
    "validation_records": len(val_df),
    "test_records": len(test_df),
    "number_of_sources": int(
        test_df["source_dataset"].nunique()
    ),
    "output_files": [
        str(metrics_path),
        str(confusion_path)
    ]
}

summary_path = OUTPUT_DIR / "phase6a_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


print("\n" + "=" * 70)
print("PHASE 6A COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nMetrics saved: {metrics_path}")
print(f"Confusion matrices saved: {confusion_path}")
print(f"Summary saved: {summary_path}")
