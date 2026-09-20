
from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PHASE 5I - CLEANED DATASET MODEL EVALUATION
# ============================================================

BASE_DIR = Path(
    r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"
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
    / "phase5"
    / "phase5i_cleaned_model_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "train": DATA_DIR / "claim_train_clean.csv",
    "validation": DATA_DIR / "claim_val_clean.csv",
    "test": DATA_DIR / "claim_test_clean.csv",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 5I - CLEANED DATASET MODEL EVALUATION")
print("=" * 70)

data = {}

for split, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "claim_text",
        "evidence_text",
        "binary_target",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split} is missing columns: {missing_columns}"
        )

    df["claim_text"] = df["claim_text"].fillna("")
    df["evidence_text"] = df["evidence_text"].fillna("")

    df["model_text"] = (
        "claim: "
        + df["claim_text"].astype(str)
        + " evidence: "
        + df["evidence_text"].astype(str)
    )

    data[split] = df

    print(
        f"{split.capitalize()} records: {len(df)}"
    )


train_df = data["train"]
val_df = data["validation"]
test_df = data["test"]


# ============================================================
# PREPARE TEXT AND LABELS
# ============================================================

X_train_text = train_df["model_text"]
X_val_text = val_df["model_text"]
X_test_text = test_df["model_text"]

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
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

print(f"Training matrix: {X_train.shape}")
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
    random_state=42,
)

model.fit(X_train, y_train)

print("Logistic Regression training completed.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, split_name):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "split": split_name,
        "records": int(len(y)),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
    }

    cm = confusion_matrix(y, predictions)

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():
        if metric not in ["split", "records"]:
            print(f"{metric}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return metrics, cm, probabilities, predictions


# ============================================================
# EVALUATE ALL SPLITS
# ============================================================

train_metrics, train_cm, _, _ = evaluate_model(
    model,
    X_train,
    y_train,
    "Train"
)

val_metrics, val_cm, _, _ = evaluate_model(
    model,
    X_val,
    y_val,
    "Validation"
)

test_metrics, test_cm, test_probabilities, test_predictions = (
    evaluate_model(
        model,
        X_test,
        y_test,
        "Cleaned Test"
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df["predicted_probability"] = test_probabilities
predictions_df["predicted_label"] = test_predictions
predictions_df["correct_prediction"] = (
    predictions_df["binary_target"].astype(int)
    == predictions_df["predicted_label"]
)

predictions_path = (
    OUTPUT_DIR
    / "cleaned_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(
    f"\nPredictions saved:\n{predictions_path}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_matrix_df = pd.DataFrame(
    test_cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

confusion_matrix_path = (
    OUTPUT_DIR
    / "cleaned_test_confusion_matrix.csv"
)

confusion_matrix_df.to_csv(
    confusion_matrix_path
)


# ============================================================
# SAVE METRICS
# ============================================================

all_metrics = pd.DataFrame([
    train_metrics,
    val_metrics,
    test_metrics,
])

metrics_path = (
    OUTPUT_DIR
    / "cleaned_model_metrics.csv"
)

all_metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "model": "TF-IDF + Logistic Regression",
    "tfidf_parameters": {
        "ngram_range": "(1, 2)",
        "min_df": 2,
        "max_features": 50000,
        "sublinear_tf": True,
    },
    "logistic_regression_parameters": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "train_records": int(len(train_df)),
    "validation_records": int(len(val_df)),
    "test_records": int(len(test_df)),
    "test_metrics": test_metrics,
    "test_confusion_matrix": test_cm.tolist(),
}

summary_path = OUTPUT_DIR / "phase5i_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5I COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Metrics saved: {metrics_path}")
print(f"Confusion matrix saved: {confusion_matrix_path}")
print(f"Summary saved: {summary_path}")
