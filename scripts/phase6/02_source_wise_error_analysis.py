from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase5"
    / "phase5h_leakage_cleaning"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6b_source_wise_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = INPUT_DIR / "claim_train_clean.csv"
VAL_PATH = INPUT_DIR / "claim_val_clean.csv"
TEST_PATH = INPUT_DIR / "claim_test_clean.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 6B - SOURCE-WISE ERROR ANALYSIS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTrain records: {len(train_df)}")
print(f"Validation records: {len(val_df)}")
print(f"Test records: {len(test_df)}")


# ============================================================
# 3. CREATE MODEL TEXT
# ============================================================

def create_model_text(df):
    claim = df["claim_text"].fillna("").astype(str)
    evidence = df["evidence_text"].fillna("").astype(str)

    return (
        "claim: " + claim +
        " evidence: " + evidence
    )


X_train_text = create_model_text(train_df)
X_val_text = create_model_text(val_df)
X_test_text = create_model_text(test_df)

y_train = train_df["binary_target"].astype(int)
y_val = val_df["binary_target"].astype(int)
y_test = test_df["binary_target"].astype(int)


# ============================================================
# 4. TF-IDF VECTORIZATION
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

print(f"Training matrix: {X_train.shape}")
print(f"Validation matrix: {X_val.shape}")
print(f"Test matrix: {X_test.shape}")


# ============================================================
# 5. TRAIN MODEL
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

print("Model training completed.")


# ============================================================
# 6. GENERATE TEST PREDICTIONS
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.50).astype(int)


# ============================================================
# 7. CONFIDENCE GROUPS
# ============================================================

def get_confidence_group(probability):

    if probability < 0.20:
        return "Very Low Probability"

    elif probability < 0.40:
        return "Low Probability"

    elif probability < 0.60:
        return "Uncertain"

    elif probability < 0.80:
        return "High Probability"

    else:
        return "Very High Probability"


# ============================================================
# 8. CREATE PREDICTION DATAFRAME
# ============================================================

predictions_df = test_df.copy()

predictions_df["predicted_label"] = test_predictions
predictions_df["probability_positive"] = test_probabilities

predictions_df["confidence_group"] = [
    get_confidence_group(probability)
    for probability in test_probabilities
]

predictions_df["correct_prediction"] = (
    predictions_df["binary_target"].astype(int)
    == predictions_df["predicted_label"]
)

predictions_df["error_type"] = "Correct"

predictions_df.loc[
    (predictions_df["binary_target"] == 0)
    & (predictions_df["predicted_label"] == 1),
    "error_type"
] = "False Positive"

predictions_df.loc[
    (predictions_df["binary_target"] == 1)
    & (predictions_df["predicted_label"] == 0),
    "error_type"
] = "False Negative"


# ============================================================
# 9. SAVE ALL TEST PREDICTIONS
# ============================================================

prediction_columns = [
    "sample_id",
    "source_dataset",
    "claim_text",
    "evidence_text",
    "standardized_label",
    "binary_target",
    "predicted_label",
    "probability_positive",
    "confidence_group",
    "correct_prediction",
    "error_type"
]

predictions_df[prediction_columns].to_csv(
    OUTPUT_DIR / "source_wise_test_predictions.csv",
    index=False
)


# ============================================================
# 10. SAVE ONLY INCORRECT PREDICTIONS
# ============================================================

errors_df = predictions_df[
    predictions_df["correct_prediction"] == False
].copy()

errors_df[prediction_columns].to_csv(
    OUTPUT_DIR / "source_wise_test_errors.csv",
    index=False
)


# ============================================================
# 11. SOURCE-WISE METRICS
# ============================================================

source_metrics = []

for source in sorted(test_df["source_dataset"].dropna().unique()):

    source_mask = (
        predictions_df["source_dataset"] == source
    )

    y_true_source = predictions_df.loc[
        source_mask, "binary_target"
    ]

    y_pred_source = predictions_df.loc[
        source_mask, "predicted_label"
    ]

    probabilities_source = predictions_df.loc[
        source_mask, "probability_positive"
    ]

    accuracy = accuracy_score(
        y_true_source,
        y_pred_source
    )

    precision = precision_score(
        y_true_source,
        y_pred_source,
        zero_division=0
    )

    recall = recall_score(
        y_true_source,
        y_pred_source,
        zero_division=0
    )

    f1 = f1_score(
        y_true_source,
        y_pred_source,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_true_source,
            probabilities_source
        )
    except ValueError:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(
        y_true_source,
        y_pred_source,
        labels=[0, 1]
    ).ravel()

    total_errors = fp + fn

    source_metrics.append({
        "source_dataset": source,
        "total_samples": len(y_true_source),
        "correct_predictions": int(
            (y_true_source == y_pred_source).sum()
        ),
        "total_errors": int(total_errors),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "mean_probability_positive": probabilities_source.mean()
    })


source_metrics_df = pd.DataFrame(source_metrics)

source_metrics_df.to_csv(
    OUTPUT_DIR / "source_wise_error_summary.csv",
    index=False
)


# ============================================================
# 12. CONFIDENCE-WISE ERROR SUMMARY
# ============================================================

confidence_summary = (
    predictions_df
    .groupby(
        ["source_dataset", "confidence_group"],
        observed=False
    )
    .agg(
        total_samples=("sample_id", "count"),
        incorrect_predictions=(
            "correct_prediction",
            lambda x: (~x).sum()
        ),
        accuracy=(
            "correct_prediction",
            "mean"
        ),
        average_probability=(
            "probability_positive",
            "mean"
        )
    )
    .reset_index()
)

confidence_summary.to_csv(
    OUTPUT_DIR / "confidence_wise_error_summary.csv",
    index=False
)


# ============================================================
# 13. ERROR TYPE SUMMARY
# ============================================================

error_type_summary = (
    predictions_df
    .groupby(
        ["source_dataset", "error_type"],
        observed=False
    )
    .size()
    .reset_index(name="count")
)

error_type_summary.to_csv(
    OUTPUT_DIR / "error_type_summary.csv",
    index=False
)


# ============================================================
# 14. SAVE SUMMARY JSON
# ============================================================

summary = {
    "phase": "6B",
    "title": "Source-Wise Error Analysis",
    "train_records": int(len(train_df)),
    "validation_records": int(len(val_df)),
    "test_records": int(len(test_df)),
    "total_test_errors": int(len(errors_df)),
    "false_positives": int(
        (predictions_df["error_type"] == "False Positive").sum()
    ),
    "false_negatives": int(
        (predictions_df["error_type"] == "False Negative").sum()
    ),
    "source_metrics": source_metrics
}

with open(
    OUTPUT_DIR / "phase6b_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4,
        default=float
    )


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-WISE ERROR SUMMARY")
print("=" * 70)

print(source_metrics_df.to_string(index=False))

print("\n" + "=" * 70)
print("ERROR TYPE COUNTS")
print("=" * 70)

print(error_type_summary.to_string(index=False))

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR}")
print("- source_wise_test_predictions.csv")
print("- source_wise_test_errors.csv")
print("- source_wise_error_summary.csv")
print("- confidence_wise_error_summary.csv")
print("- error_type_summary.csv")
print("- phase6b_summary.json")

print("\nPhase 6B completed successfully.")