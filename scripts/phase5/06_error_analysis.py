import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection"

TRAIN_PATH = "data/processed/phase5/datasets/claim_train.csv"
VAL_PATH = "data/processed/phase5/datasets/claim_val.csv"
TEST_PATH = "data/processed/phase5/datasets/claim_test.csv"

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "phase5", "phase5f_error_analysis"
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

    return "Claim: " + claim + " Evidence: " + evidence


X_train_text = prepare_text(train_df)
X_val_text = prepare_text(val_df)
X_test_text = prepare_text(test_df)

y_train = train_df["binary_target"].astype(int)
y_test = test_df["binary_target"].astype(int)


# ============================================================
# 4. TF-IDF
# ============================================================

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF training shape:", X_train.shape)
print("TF-IDF test shape:", X_test.shape)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]

# Primary threshold
threshold = 0.50

test_predictions = (
    test_probabilities >= threshold
).astype(int)


# ============================================================
# 7. CREATE ERROR ANALYSIS DATAFRAME
# ============================================================

analysis_df = test_df.copy()

analysis_df["predicted_probability"] = test_probabilities
analysis_df["predicted_label"] = test_predictions

analysis_df["correct_prediction"] = (
    analysis_df["binary_target"] ==
    analysis_df["predicted_label"]
)

analysis_df["error_type"] = "Correct"

analysis_df.loc[
    (analysis_df["binary_target"] == 0) &
    (analysis_df["predicted_label"] == 1),
    "error_type"
] = "False Positive"

analysis_df.loc[
    (analysis_df["binary_target"] == 1) &
    (analysis_df["predicted_label"] == 0),
    "error_type"
] = "False Negative"


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_df["binary_target"],
    test_predictions
)

print("\nConfusion Matrix")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nError Counts")
print("-" * 40)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)


# ============================================================
# 9. SAVE FALSE POSITIVES
# ============================================================

false_positives = analysis_df[
    analysis_df["error_type"] == "False Positive"
].copy()

false_positives = false_positives.sort_values(
    by="predicted_probability",
    ascending=False
)

fp_path = os.path.join(
    OUTPUT_DIR,
    "false_positives.csv"
)

false_positives.to_csv(
    fp_path,
    index=False
)


# ============================================================
# 10. SAVE FALSE NEGATIVES
# ============================================================

false_negatives = analysis_df[
    analysis_df["error_type"] == "False Negative"
].copy()

false_negatives = false_negatives.sort_values(
    by="predicted_probability",
    ascending=True
)

fn_path = os.path.join(
    OUTPUT_DIR,
    "false_negatives.csv"
)

false_negatives.to_csv(
    fn_path,
    index=False
)


# ============================================================
# 11. LOW-CONFIDENCE PREDICTIONS
# ============================================================

# Distance from the decision threshold
analysis_df["confidence_distance"] = abs(
    analysis_df["predicted_probability"] - threshold
)

low_confidence = analysis_df.sort_values(
    by="confidence_distance",
    ascending=True
).head(100)

low_confidence_path = os.path.join(
    OUTPUT_DIR,
    "low_confidence_predictions_top100.csv"
)

low_confidence.to_csv(
    low_confidence_path,
    index=False
)


# ============================================================
# 12. ALL PREDICTIONS
# ============================================================

all_predictions_path = os.path.join(
    OUTPUT_DIR,
    "all_test_predictions_with_errors.csv"
)

analysis_df.to_csv(
    all_predictions_path,
    index=False
)


# ============================================================
# 13. ERROR DISTRIBUTION
# ============================================================

error_distribution = (
    analysis_df["error_type"]
    .value_counts()
    .reset_index()
)

error_distribution.columns = [
    "error_type",
    "count"
]

error_distribution_path = os.path.join(
    OUTPUT_DIR,
    "error_distribution.csv"
)

error_distribution.to_csv(
    error_distribution_path,
    index=False
)

print("\nError Distribution")
print(error_distribution.to_string(index=False))


# ============================================================
# 14. CONFIDENCE GROUP ANALYSIS
# ============================================================

def confidence_group(probability):
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


analysis_df["confidence_group"] = (
    analysis_df["predicted_probability"]
    .apply(confidence_group)
)

confidence_analysis = (
    analysis_df.groupby("confidence_group")
    .agg(
        total_samples=("sample_id", "count"),
        incorrect_predictions=(
            "correct_prediction",
            lambda x: (~x).sum()
        ),
        accuracy=("correct_prediction", "mean")
    )
    .reset_index()
)

confidence_analysis_path = os.path.join(
    OUTPUT_DIR,
    "confidence_group_analysis.csv"
)

confidence_analysis.to_csv(
    confidence_analysis_path,
    index=False
)

print("\nConfidence Group Analysis")
print(confidence_analysis.to_string(index=False))


# ============================================================
# 15. PRINT SAMPLE ERRORS
# ============================================================

print("\nSample False Positives")
print("-" * 60)

print(
    false_positives[
        [
            "sample_id",
            "claim_text",
            "evidence_text",
            "predicted_probability"
        ]
    ].head(5).to_string(index=False)
)

print("\nSample False Negatives")
print("-" * 60)

print(
    false_negatives[
        [
            "sample_id",
            "claim_text",
            "evidence_text",
            "predicted_probability"
        ]
    ].head(5).to_string(index=False)
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\nFiles saved successfully:")
print(fp_path)
print(fn_path)
print(low_confidence_path)
print(all_predictions_path)
print(error_distribution_path)
print(confidence_analysis_path)

print("\nPhase 5F completed successfully.")