
from pathlib import Path
import pandas as pd
import json


# ============================================================
# PHASE 6E: RESEARCH FINDINGS AND INTERPRETATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase6"
    / "phase6d_manual_inspection_visualization"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase6"
    / "phase6e_research_findings"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    path = INPUT_DIR / filename

    if not path.exists():
        print(f"WARNING: Missing file: {filename}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"Loaded: {filename} | Shape: {df.shape}")
    return df


# ------------------------------------------------------------
# Load Phase 6D outputs
# ------------------------------------------------------------

source_summary = load_csv("phase6d_source_wise_summary.csv")
error_type_summary = load_csv("phase6d_error_type_summary.csv")
pattern_summary = load_csv("qualitative_error_pattern_summary.csv")
confidence_summary = load_csv("confidence_error_rate_summary.csv")
high_confidence = load_csv("phase6d_high_confidence_errors.csv")


# ------------------------------------------------------------
# Extract key statistics
# ------------------------------------------------------------

findings = []

if not source_summary.empty:

    for _, row in source_summary.iterrows():

        source = row.get("source_dataset", "Unknown")
        total = int(row.get("total_samples", 0))
        errors = int(row.get("total_errors", 0))
        accuracy = float(row.get("accuracy", 0))
        precision = float(row.get("precision", 0))
        recall = float(row.get("recall", 0))
        f1 = float(row.get("f1_score", 0))

        findings.append({
            "source": source,
            "total_samples": total,
            "total_errors": errors,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })


# ------------------------------------------------------------
# Generate interpretation
# ------------------------------------------------------------

report_lines = []

report_lines.append("# Phase 6E: Research Findings and Interpretation\n")

report_lines.append(
    "## 1. Overview\n"
)

report_lines.append(
    "Phase 6E interprets the source-wise evaluation, error analysis, "
    "confidence analysis, and qualitative error patterns generated "
    "during Phase 6A–6D.\n"
)

report_lines.append(
    "The analysis focuses on identifying model limitations, "
    "source-specific behavior, and potential directions for improvement.\n"
)


# ------------------------------------------------------------
# Source-wise findings
# ------------------------------------------------------------

report_lines.append("## 2. Source-wise Performance Findings\n")

if not source_summary.empty:

    for _, row in source_summary.iterrows():

        source = row["source_dataset"]
        total = int(row["total_samples"])
        errors = int(row["total_errors"])
        fp = int(row["false_positives"])
        fn = int(row["false_negatives"])
        accuracy = float(row["accuracy"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        f1 = float(row["f1_score"])

        report_lines.append(f"### {source}\n")

        report_lines.append(
            f"- Total test samples: {total}\n"
            f"- Total errors: {errors}\n"
            f"- False positives: {fp}\n"
            f"- False negatives: {fn}\n"
            f"- Accuracy: {accuracy:.4f}\n"
            f"- Precision: {precision:.4f}\n"
            f"- Recall: {recall:.4f}\n"
            f"- F1-score: {f1:.4f}\n"
        )

        if fn > fp:
            report_lines.append(
                "The source shows more false negatives than false positives, "
                "indicating that some misleading claims may be classified "
                "as supported.\n"
            )

        elif fp > fn:
            report_lines.append(
                "The source shows more false positives than false negatives, "
                "indicating that some supported claims may be classified "
                "as misleading.\n"
            )

        else:
            report_lines.append(
                "False positives and false negatives occur at similar levels "
                "for this source.\n"
            )


# ------------------------------------------------------------
# Error pattern findings
# ------------------------------------------------------------

report_lines.append("## 3. Qualitative Error Pattern Findings\n")

if not pattern_summary.empty:

    pattern_col = "error_pattern"
    count_col = "count"

    if pattern_col in pattern_summary.columns:

        sorted_patterns = pattern_summary.sort_values(
            by=count_col,
            ascending=False
        )

        for _, row in sorted_patterns.iterrows():

            pattern = row[pattern_col]
            count = int(row[count_col])

            report_lines.append(
                f"- **{pattern}**: {count} errors identified.\n"
            )

        most_common = sorted_patterns.iloc[0][pattern_col]

        report_lines.append(
            f"\nThe most frequently identified qualitative pattern was "
            f"**{most_common}**. These patterns are heuristic indicators "
            f"and should not be interpreted as proof of causal relationships.\n"
        )

else:

    report_lines.append(
        "The qualitative pattern summary was unavailable.\n"
    )


# ------------------------------------------------------------
# Confidence findings
# ------------------------------------------------------------

report_lines.append("## 4. Confidence-based Findings\n")

if not confidence_summary.empty:

    report_lines.append(
        "Confidence-based analysis was used to identify cases where "
        "the model made incorrect predictions with varying confidence levels.\n"
    )

    if "accuracy" in confidence_summary.columns:

        lowest_accuracy = confidence_summary.loc[
            confidence_summary["accuracy"].idxmin()
        ]

        report_lines.append(
            f"- Lowest observed confidence-group accuracy: "
            f"{lowest_accuracy['accuracy']:.4f}\n"
        )

        report_lines.append(
            f"- Corresponding confidence group: "
            f"{lowest_accuracy.get('confidence_group', 'Unknown')}\n"
        )

else:

    report_lines.append(
        "Confidence summary was unavailable.\n"
    )


# ------------------------------------------------------------
# High-confidence errors
# ------------------------------------------------------------

report_lines.append("## 5. High-confidence Incorrect Predictions\n")

if not high_confidence.empty:

    report_lines.append(
        f"The model produced **{len(high_confidence)} high-confidence "
        f"incorrect predictions** in the inspected error set.\n"
    )

    report_lines.append(
        "These examples are important for further manual review because "
        "high confidence does not necessarily indicate correct classification.\n"
    )

else:

    report_lines.append(
        "No high-confidence error records were available.\n"
    )


# ------------------------------------------------------------
# Limitations
# ------------------------------------------------------------

report_lines.append("## 6. Identified Model Limitations\n")

limitations = [
    "The model relies primarily on textual features and may not fully capture visual infographic information.",
    "False positives indicate that supported claims can be classified as misleading.",
    "False negatives indicate that misleading claims can be classified as supported.",
    "Source-wise performance may be unstable for sources with relatively few test samples.",
    "Heuristic linguistic patterns do not establish that a particular language pattern causes classification errors.",
    "High-confidence incorrect predictions indicate that model confidence should not be treated as a guarantee of correctness.",
    "The model may have difficulty interpreting context, uncertainty, numerical claims, and causal language."
]

for limitation in limitations:
    report_lines.append(f"- {limitation}\n")


# ------------------------------------------------------------
# Future improvements
# ------------------------------------------------------------

report_lines.append("## 7. Potential Future Improvements\n")

improvements = [
    "Incorporate visual features from infographic images.",
    "Investigate multimodal fusion of image and text representations.",
    "Perform deeper manual annotation of false positives and false negatives.",
    "Evaluate transformer-based text encoders.",
    "Apply calibration methods to improve probability reliability.",
    "Perform additional cross-source and cross-domain validation.",
    "Use explainability methods to investigate important textual and visual features."
]

for improvement in improvements:
    report_lines.append(f"- {improvement}\n")


# ------------------------------------------------------------
# Save Markdown report
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "phase6e_research_findings_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))


# ------------------------------------------------------------
# Save JSON summary
# ------------------------------------------------------------

summary = {
    "phase": "6E",
    "description": "Research findings and interpretation",
    "source_wise_findings": findings,
    "high_confidence_error_count": (
        int(len(high_confidence))
        if not high_confidence.empty
        else 0
    ),
    "limitations_identified": limitations,
    "potential_improvements": improvements
}

summary_path = OUTPUT_DIR / "phase6e_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


print("\n" + "=" * 65)
print("PHASE 6E COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Report saved: {report_path}")
print(f"Summary saved: {summary_path}")
