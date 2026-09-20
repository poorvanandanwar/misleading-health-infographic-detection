import os
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. File paths
# =====================================================

TRAIN_PATH = "data/processed/phase5/datasets/claim_train.csv"
VAL_PATH = "data/processed/phase5/datasets/claim_val.csv"
TEST_PATH = "data/processed/phase5/datasets/claim_test.csv"

RESULTS_DIR = "results/phase5/tfidf_baseline"
MODEL_DIR = "models/phase5"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# 2. Load datasets
# =====================================================

print("Loading datasets...")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# =====================================================
# 3. Extract inputs and labels
# =====================================================

X_train = train["model_text"].fillna("")
X_val = val["model_text"].fillna("")
X_test = test["model_text"].fillna("")

y_train = train["binary_target"]
y_val = val["binary_target"]
y_test = test["binary_target"]


# =====================================================
# 4. TF-IDF Vectorization
# Fit ONLY on training data
# =====================================================

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Validation TF-IDF shape:", X_val_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)


# =====================================================
# 5. Train Logistic Regression
# =====================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# =====================================================
# 6. Evaluation function
# =====================================================

def evaluate_model(X, y, dataset_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
        "auroc": roc_auc_score(y, probabilities)
    }

    print(f"\n{dataset_name} Results")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    return metrics, predictions, probabilities


# =====================================================
# 7. Evaluate train, validation and test
# =====================================================

train_metrics, _, _ = evaluate_model(
    X_train_tfidf, y_train, "Training"
)

val_metrics, _, _ = evaluate_model(
    X_val_tfidf, y_val, "Validation"
)

test_metrics, test_predictions, test_probabilities = evaluate_model(
    X_test_tfidf, y_test, "Test"
)


# =====================================================
# 8. Save metrics
# =====================================================

all_metrics = {
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics
}

metrics_path = os.path.join(
    RESULTS_DIR, "metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(all_metrics, file, indent=4)


# =====================================================
# 9. Save test predictions
# =====================================================

predictions_df = pd.DataFrame({
    "sample_id": test["sample_id"],
    "actual_label": y_test,
    "predicted_label": test_predictions,
    "prediction_probability": test_probabilities
})

predictions_path = os.path.join(
    RESULTS_DIR, "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# =====================================================
# 10. Save classification report
# =====================================================

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

report_path = os.path.join(
    RESULTS_DIR, "classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)


# =====================================================
# 11. Save trained model and vectorizer
# =====================================================

model_path = os.path.join(
    MODEL_DIR,
    "tfidf_logistic_regression.joblib"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)


print("\n" + "=" * 50)
print("PHASE 5B COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Metrics saved to:", metrics_path)
print("Predictions saved to:", predictions_path)
print("Model saved to:", model_path)
print("Vectorizer saved to:", vectorizer_path)