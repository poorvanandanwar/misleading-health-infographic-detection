import os
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
from joblib import dump


# ==============================
# PATHS
# ==============================

TRAIN_PATH = "data/processed/phase5/datasets/claim_train.csv"
VAL_PATH = "data/processed/phase5/datasets/claim_val.csv"
TEST_PATH = "data/processed/phase5/datasets/claim_test.csv"

RESULTS_DIR = "results/phase5/hyperparameter_experiments"
MODELS_DIR = "models/phase5/hyperparameter_experiments"

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

def create_combined_text(df):
    claim = df["claim_text"].fillna("").astype(str)
    evidence = df["evidence_text"].fillna("").astype(str)

    return (
        "Claim: " + claim +
        " Evidence: " + evidence
    )


X_train_text = create_combined_text(train_df)
X_val_text = create_combined_text(val_df)
X_test_text = create_combined_text(test_df)


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
    {
        "name": "unigram_balanced",
        "ngram_range": (1, 1),
        "class_weight": "balanced"
    },
    {
        "name": "bigram_balanced",
        "ngram_range": (1, 2),
        "class_weight": "balanced"
    },
    {
        "name": "trigram_balanced",
        "ngram_range": (1, 3),
        "class_weight": "balanced"
    },
    {
        "name": "unigram_no_weight",
        "ngram_range": (1, 1),
        "class_weight": None
    },
    {
        "name": "bigram_no_weight",
        "ngram_range": (1, 2),
        "class_weight": None
    },
    {
        "name": "trigram_no_weight",
        "ngram_range": (1, 3),
        "class_weight": None
    }
]


all_results = []


# ==============================
# RUN EXPERIMENTS
# ==============================

for config in experiments:

    experiment_name = config["name"]
    ngram_range = config["ngram_range"]
    class_weight = config["class_weight"]

    print("\n" + "=" * 70)
    print("Running experiment:", experiment_name)
    print("N-gram range:", ngram_range)
    print("Class weight:", class_weight)
    print("=" * 70)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    # Fit only on training data
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Testing shape:", X_test.shape)

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
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

    # Print metrics
    print("\nValidation Results")
    print("Accuracy:", round(val_accuracy, 4))
    print("Precision:", round(val_precision, 4))
    print("Recall:", round(val_recall, 4))
    print("F1:", round(val_f1, 4))
    print("AUROC:", round(val_auroc, 4))

    print("\nTest Results")
    print("Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(test_precision, 4))
    print("Recall:", round(test_recall, 4))
    print("F1:", round(test_f1, 4))
    print("AUROC:", round(test_auroc, 4))

    print("\nTest Confusion Matrix")
    print(confusion_matrix(y_test, test_predictions))

    # Save predictions
    predictions_df = test_df.copy()

    predictions_df["prediction"] = test_predictions
    predictions_df["probability"] = test_probabilities

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{experiment_name}_test_predictions.csv"
    )

    predictions_df.to_csv(predictions_path, index=False)

    # Save model
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
    all_results.append({
        "experiment": experiment_name,
        "ngram_range": str(ngram_range),
        "class_weight": str(class_weight),

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
    })


# ==============================
# SAVE RESULTS
# ==============================

results_df = pd.DataFrame(all_results)

csv_path = os.path.join(
    RESULTS_DIR,
    "hyperparameter_comparison.csv"
)

results_df.to_csv(csv_path, index=False)

json_path = os.path.join(
    RESULTS_DIR,
    "hyperparameter_comparison.json"
)

with open(json_path, "w") as file:
    json.dump(all_results, file, indent=4)


# ==============================
# DISPLAY FINAL RESULTS
# ==============================

print("\n" + "=" * 90)
print("FINAL HYPERPARAMETER COMPARISON")
print("=" * 90)

display_columns = [
    "experiment",
    "ngram_range",
    "class_weight",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auroc"
]

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nResults saved to:")
print(csv_path)

print("\nModels saved to:")
print(MODELS_DIR)

print("\nPhase 5D hyperparameter experiments completed successfully.")