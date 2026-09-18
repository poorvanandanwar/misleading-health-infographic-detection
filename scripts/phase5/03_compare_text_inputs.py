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
from joblib import dump


# ==============================
# PATHS
# ==============================

TRAIN_PATH = "data/processed/phase5/datasets/claim_train.csv"
VAL_PATH = "data/processed/phase5/datasets/claim_val.csv"
TEST_PATH = "data/processed/phase5/datasets/claim_test.csv"

RESULTS_DIR = "results/phase5/text_input_comparison"
MODELS_DIR = "models/phase5/text_input_comparison"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ==============================
# PREPARE TEXT
# ==============================

def clean_text(series):
    return series.fillna("").astype(str)


def create_text_variants(df):
    claim_text = clean_text(df["claim_text"])
    evidence_text = clean_text(df["evidence_text"])

    claim_only = (
        "Claim: " + claim_text
    )

    evidence_only = (
        "Evidence: " + evidence_text
    )

    claim_evidence = (
        "Claim: " + claim_text +
        " Evidence: " + evidence_text
    )

    return {
        "claim_only": claim_only,
        "evidence_only": evidence_only,
        "claim_evidence": claim_evidence
    }


train_texts = create_text_variants(train_df)
val_texts = create_text_variants(val_df)
test_texts = create_text_variants(test_df)


# ==============================
# TARGET LABELS
# ==============================

y_train = train_df["binary_target"]
y_val = val_df["binary_target"]
y_test = test_df["binary_target"]


# ==============================
# EXPERIMENT CONFIGURATION
# ==============================

experiments = [
    "claim_only",
    "evidence_only",
    "claim_evidence"
]


all_results = []


# ==============================
# TRAIN AND EVALUATE
# ==============================

for experiment_name in experiments:

    print("\n" + "=" * 60)
    print("Running experiment:", experiment_name)
    print("=" * 60)

    X_train_text = train_texts[experiment_name]
    X_val_text = val_texts[experiment_name]
    X_test_text = test_texts[experiment_name]

    # Fit vectorizer ONLY on training data
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training TF-IDF shape:", X_train.shape)
    print("Validation TF-IDF shape:", X_val.shape)
    print("Testing TF-IDF shape:", X_test.shape)

    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Validation predictions
    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Test predictions
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Validation metrics
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    val_recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    val_f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )
    val_auroc = roc_auc_score(y_val, val_probabilities)

    # Test metrics
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    test_auroc = roc_auc_score(y_test, test_probabilities)

    # Print results
    print("\nValidation Results")
    print("Accuracy:", round(val_accuracy, 4))
    print("Precision:", round(val_precision, 4))
    print("Recall:", round(val_recall, 4))
    print("F1 Score:", round(val_f1, 4))
    print("AUROC:", round(val_auroc, 4))

    print("\nTest Results")
    print("Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(test_precision, 4))
    print("Recall:", round(test_recall, 4))
    print("F1 Score:", round(test_f1, 4))
    print("AUROC:", round(test_auroc, 4))

    print("\nTest Confusion Matrix")
    print(confusion_matrix(y_test, test_predictions))

    print("\nTest Classification Report")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    # Save predictions
    predictions_df = test_df.copy()

    predictions_df["prediction"] = test_predictions
    predictions_df["probability"] = test_probabilities

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{experiment_name}_test_predictions.csv"
    )

    predictions_df.to_csv(predictions_path, index=False)

    # Save model and vectorizer
    model_path = os.path.join(
        MODELS_DIR,
        f"{experiment_name}_logistic_regression.joblib"
    )

    vectorizer_path = os.path.join(
        MODELS_DIR,
        f"{experiment_name}_vectorizer.joblib"
    )

    dump(model, model_path)
    dump(vectorizer, vectorizer_path)

    # Store results
    result = {
        "experiment": experiment_name,

        "validation_accuracy": val_accuracy,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,
        "validation_auroc": val_auroc,

        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auroc": test_auroc,

        "number_of_features": X_train.shape[1]
    }

    all_results.append(result)


# ==============================
# SAVE COMPARISON RESULTS
# ==============================

results_df = pd.DataFrame(all_results)

results_path = os.path.join(
    RESULTS_DIR,
    "experiment_comparison.csv"
)

results_df.to_csv(results_path, index=False)

json_path = os.path.join(
    RESULTS_DIR,
    "experiment_comparison.json"
)

with open(json_path, "w") as file:
    json.dump(all_results, file, indent=4)


# ==============================
# DISPLAY FINAL COMPARISON
# ==============================

print("\n" + "=" * 80)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 80)

display_columns = [
    "experiment",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auroc"
]

print(
    results_df[display_columns].round(4).to_string(index=False)
)

print("\nResults saved to:")
print(results_path)

print("\nModels saved to:")
print(MODELS_DIR)

print("\nPhase 5C text input comparison completed successfully.")