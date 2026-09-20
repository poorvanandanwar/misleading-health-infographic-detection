import os
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
    classification_report
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"

TRAIN_PATH = "data/processed/phase5/datasets/claim_train.csv"
VAL_PATH = "data/processed/phase5/datasets/claim_val.csv"
TEST_PATH = "data/processed/phase5/datasets/claim_test.csv"

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "phase5"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 3. PREPARE TEXT
# ============================================================

def prepare_text(df):
    claim = df["claim_text"].fillna("").astype(str)
    evidence = df["evidence_text"].fillna("").astype(str)

    return (
        "Claim: " + claim +
        " Evidence: " + evidence
    )


X_train_text = prepare_text(train_df)
X_val_text = prepare_text(val_df)
X_test_text = prepare_text(test_df)

y_train = train_df["binary_target"].astype(int)
y_val = val_df["binary_target"].astype(int)
y_test = test_df["binary_target"].astype(int)


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

print("Train TF-IDF shape:", X_train.shape)
print("Validation TF-IDF shape:", X_val.shape)
print("Test TF-IDF shape:", X_test.shape)


# ============================================================
# 5. TRAIN BASELINE MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 6. GET VALIDATION AND TEST PROBABILITIES
# ============================================================

val_probabilities = model.predict_proba(X_val)[:, 1]
test_probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. THRESHOLD TUNING ON VALIDATION SET
# ============================================================

thresholds = np.arange(0.30, 0.71, 0.05)

validation_results = []

print("\nValidation Threshold Results")
print("-" * 75)

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_val, val_predictions)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    validation_results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


validation_results_df = pd.DataFrame(validation_results)

validation_results_path = os.path.join(
    OUTPUT_DIR,
    "phase5e_validation_threshold_results.csv"
)

validation_results_df.to_csv(
    validation_results_path,
    index=False
)


# ============================================================
# 8. SELECT BEST THRESHOLD USING VALIDATION F1
# ============================================================

best_row = validation_results_df.loc[
    validation_results_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nBest Threshold Selected")
print("-" * 40)
print(f"Best threshold: {best_threshold:.2f}")
print(f"Validation accuracy: {best_row['accuracy']:.4f}")
print(f"Validation precision: {best_row['precision']:.4f}")
print(f"Validation recall: {best_row['recall']:.4f}")
print(f"Validation F1-score: {best_row['f1_score']:.4f}")


# ============================================================
# 9. EVALUATE DEFAULT THRESHOLD ON TEST SET
# ============================================================

default_threshold = 0.50

test_predictions_default = (
    test_probabilities >= default_threshold
).astype(int)

default_accuracy = accuracy_score(
    y_test,
    test_predictions_default
)

default_precision = precision_score(
    y_test,
    test_predictions_default,
    zero_division=0
)

default_recall = recall_score(
    y_test,
    test_predictions_default,
    zero_division=0
)

default_f1 = f1_score(
    y_test,
    test_predictions_default,
    zero_division=0
)

default_auc = roc_auc_score(
    y_test,
    test_probabilities
)


# ============================================================
# 10. EVALUATE TUNED THRESHOLD ON TEST SET
# ============================================================

test_predictions_tuned = (
    test_probabilities >= best_threshold
).astype(int)

tuned_accuracy = accuracy_score(
    y_test,
    test_predictions_tuned
)

tuned_precision = precision_score(
    y_test,
    test_predictions_tuned,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    test_predictions_tuned,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    test_predictions_tuned,
    zero_division=0
)


# ============================================================
# 11. PRINT COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Default_Threshold_0.50": [
        default_accuracy,
        default_precision,
        default_recall,
        default_f1,
        default_auc
    ],
    "Tuned_Threshold": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        default_auc
    ]
})

print("\nTest Set Comparison")
print("-" * 75)
print(comparison_df.to_string(index=False))


comparison_path = os.path.join(
    OUTPUT_DIR,
    "phase5e_test_threshold_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions_tuned
)

print("\nConfusion Matrix - Tuned Threshold")
print(cm)

print("\nClassification Report - Tuned Threshold")
print(
    classification_report(
        y_test,
        test_predictions_tuned,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df["predicted_probability"] = test_probabilities

predictions_df["prediction_default_0.50"] = (
    test_predictions_default
)

predictions_df["prediction_tuned_threshold"] = (
    test_predictions_tuned
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "phase5e_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 14. SAVE SUMMARY JSON
# ============================================================

summary = {
    "model": "Logistic Regression",
    "text_input": "claim + evidence",
    "ngram_range": "(1,2)",
    "class_weight": "balanced",
    "default_threshold": 0.50,
    "selected_threshold": best_threshold,
    "validation_best_f1": float(best_row["f1_score"]),
    "default_test_metrics": {
        "accuracy": default_accuracy,
        "precision": default_precision,
        "recall": default_recall,
        "f1_score": default_f1,
        "roc_auc": default_auc
    },
    "tuned_test_metrics": {
        "accuracy": tuned_accuracy,
        "precision": tuned_precision,
        "recall": tuned_recall,
        "f1_score": tuned_f1,
        "roc_auc": default_auc
    },
    "confusion_matrix": cm.tolist()
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "phase5e_summary.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 15. FINAL OUTPUT LOCATIONS
# ============================================================

print("\nFiles saved successfully:")
print(validation_results_path)
print(comparison_path)
print(predictions_path)
print(summary_path)

print("\nPhase 5E completed successfully.")