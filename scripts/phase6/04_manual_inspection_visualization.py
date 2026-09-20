
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE6B_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6b_source_wise_error_analysis"
)

PHASE6C_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6c_qualitative_error_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase6"
    / "phase6d_manual_inspection_visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    """
    Save plot as PNG with good resolution.
    """
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def save_csv(dataframe, filename):
    """
    Save dataframe as CSV.
    """
    output_path = OUTPUT_DIR / filename
    dataframe.to_csv(output_path, index=False)

    print(f"Saved CSV: {output_path}")


def safe_read_csv(path):
    """
    Read CSV if available.
    """
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None

    try:
        dataframe = pd.read_csv(path)
        print(f"Loaded: {path.name} | Shape: {dataframe.shape}")
        return dataframe

    except Exception as error:
        print(f"ERROR reading {path}: {error}")
        return None


# ============================================================
# 3. LOAD PHASE 6B AND 6C OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6D: MANUAL INSPECTION AND VISUALIZATION")
print("=" * 70)

error_summary_path = (
    PHASE6B_DIR / "source_wise_error_summary.csv"
)

confidence_summary_path = (
    PHASE6B_DIR / "confidence_wise_error_summary.csv"
)

error_type_summary_path = (
    PHASE6B_DIR / "error_type_summary.csv"
)

qualitative_pattern_path = (
    PHASE6C_DIR / "qualitative_error_pattern_summary.csv"
)

overall_pattern_path = (
    PHASE6C_DIR / "overall_error_pattern_counts.csv"
)

representative_errors_path = (
    PHASE6C_DIR / "representative_error_examples.csv"
)

high_confidence_path = (
    PHASE6C_DIR / "high_confidence_errors.csv"
)

text_statistics_path = (
    PHASE6C_DIR / "error_text_statistics.csv"
)


source_error_summary = safe_read_csv(error_summary_path)
confidence_summary = safe_read_csv(confidence_summary_path)
error_type_summary = safe_read_csv(error_type_summary_path)
qualitative_patterns = safe_read_csv(qualitative_pattern_path)
overall_patterns = safe_read_csv(overall_pattern_path)
representative_errors = safe_read_csv(representative_errors_path)
high_confidence_errors = safe_read_csv(high_confidence_path)
text_statistics = safe_read_csv(text_statistics_path)


# ============================================================
# 4. VISUALIZATION 1:
# ERROR PATTERN DISTRIBUTION
# ============================================================

pattern_data = None

if overall_patterns is not None and len(overall_patterns) > 0:
    pattern_data = overall_patterns.copy()

elif qualitative_patterns is not None and len(qualitative_patterns) > 0:
    pattern_data = qualitative_patterns.copy()


if pattern_data is not None:

    print("\nGenerating error pattern distribution plot...")

    print("Pattern columns:", list(pattern_data.columns))

    pattern_column = None
    count_column = None

    possible_pattern_columns = [
        "pattern",
        "error_pattern",
        "qualitative_pattern",
        "Pattern",
        "Error Pattern"
    ]

    possible_count_columns = [
        "count",
        "error_count",
        "total",
        "Count",
        "Frequency"
    ]

    for column in possible_pattern_columns:
        if column in pattern_data.columns:
            pattern_column = column
            break

    for column in possible_count_columns:
        if column in pattern_data.columns:
            count_column = column
            break

    if pattern_column is not None and count_column is not None:

        plot_data = pattern_data.sort_values(
            by=count_column,
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_data[pattern_column].astype(str),
            plot_data[count_column]
        )

        plt.xlabel("Number of Errors")
        plt.ylabel("Error Pattern")
        plt.title("Distribution of Qualitative Error Patterns")

        save_plot("error_pattern_distribution.png")

    else:
        print(
            "WARNING: Could not identify pattern/count columns."
        )


# ============================================================
# 5. VISUALIZATION 2:
# SOURCE-WISE ERROR COMPARISON
# ============================================================

if source_error_summary is not None:

    print("\nGenerating source-wise error comparison plot...")

    print(
        "Source summary columns:",
        list(source_error_summary.columns)
    )

    source_column = None
    error_count_column = None

    possible_source_columns = [
        "source_dataset",
        "source",
        "Source",
        "dataset"
    ]

    possible_error_columns = [
        "errors",
        "error_count",
        "total_errors",
        "Errors"
    ]

    for column in possible_source_columns:
        if column in source_error_summary.columns:
            source_column = column
            break

    for column in possible_error_columns:
        if column in source_error_summary.columns:
            error_count_column = column
            break

    if source_column is not None and error_count_column is not None:

        plt.figure(figsize=(8, 5))

        plt.bar(
            source_error_summary[source_column].astype(str),
            source_error_summary[error_count_column]
        )

        plt.xlabel("Source Dataset")
        plt.ylabel("Number of Errors")
        plt.title("Source-wise Error Comparison")

        plt.xticks(rotation=20)

        save_plot("source_error_comparison.png")

    else:
        print(
            "WARNING: Could not identify source/error columns."
        )


# ============================================================
# 6. VISUALIZATION 3:
# FALSE POSITIVE VS FALSE NEGATIVE BY SOURCE
# ============================================================

if source_error_summary is not None:

    print("\nGenerating error type by source plot...")

    source_column = None

    for column in [
        "source_dataset",
        "source",
        "Source",
        "dataset"
    ]:
        if column in source_error_summary.columns:
            source_column = column
            break

    fp_column = None
    fn_column = None

    for column in [
        "FP",
        "false_positives",
        "false_positive",
        "False Positives"
    ]:
        if column in source_error_summary.columns:
            fp_column = column
            break

    for column in [
        "FN",
        "false_negatives",
        "false_negative",
        "False Negatives"
    ]:
        if column in source_error_summary.columns:
            fn_column = column
            break

    if (
        source_column is not None
        and fp_column is not None
        and fn_column is not None
    ):

        plot_data = source_error_summary.set_index(
            source_column
        )[[fp_column, fn_column]]

        plot_data.columns = [
            "False Positives",
            "False Negatives"
        ]

        plt.figure(figsize=(8, 5))

        plot_data.plot(
            kind="bar",
            figsize=(8, 5)
        )

        plt.xlabel("Source Dataset")
        plt.ylabel("Number of Errors")
        plt.title("False Positives and False Negatives by Source")
        plt.xticks(rotation=20)
        plt.legend(title="Error Type")

        save_plot("error_type_by_source.png")

    else:
        print(
            "WARNING: Could not identify FP/FN columns."
        )


# ============================================================
# 7. VISUALIZATION 4:
# CONFIDENCE-WISE ERROR DISTRIBUTION
# ============================================================

if confidence_summary is not None:

    print("\nGenerating confidence-wise error plot...")

    print(
        "Confidence summary columns:",
        list(confidence_summary.columns)
    )

    confidence_column = None
    total_column = None
    incorrect_column = None

    for column in [
        "confidence_group",
        "confidence",
        "Confidence Group",
        "probability_group"
    ]:
        if column in confidence_summary.columns:
            confidence_column = column
            break

    for column in [
        "total",
        "total_samples",
        "count",
        "Total"
    ]:
        if column in confidence_summary.columns:
            total_column = column
            break

    for column in [
        "incorrect",
        "incorrect_predictions",
        "errors",
        "Incorrect"
    ]:
        if column in confidence_summary.columns:
            incorrect_column = column
            break

    if (
        confidence_column is not None
        and total_column is not None
        and incorrect_column is not None
    ):

        plot_data = confidence_summary.copy()

        plot_data["error_rate"] = (
            plot_data[incorrect_column]
            / plot_data[total_column]
        ) * 100

        plt.figure(figsize=(10, 6))

        plt.bar(
            plot_data[confidence_column].astype(str),
            plot_data["error_rate"]
        )

        plt.xlabel("Confidence Group")
        plt.ylabel("Error Rate (%)")
        plt.title("Error Rate Across Confidence Groups")

        plt.xticks(rotation=30)

        save_plot("confidence_error_distribution.png")

        save_csv(
            plot_data,
            "confidence_error_rate_summary.csv"
        )

    else:
        print(
            "WARNING: Could not identify confidence columns."
        )


# ============================================================
# 8. SAVE ERROR TYPE SUMMARY
# ============================================================

if error_type_summary is not None:

    save_csv(
        error_type_summary,
        "phase6d_error_type_summary.csv"
    )


# ============================================================
# 9. SAVE SOURCE SUMMARY
# ============================================================

if source_error_summary is not None:

    save_csv(
        source_error_summary,
        "phase6d_source_wise_summary.csv"
    )


# ============================================================
# 10. SAVE HIGH-CONFIDENCE ERRORS
# ============================================================

if high_confidence_errors is not None:

    save_csv(
        high_confidence_errors,
        "phase6d_high_confidence_errors.csv"
    )


# ============================================================
# 11. CREATE MANUAL INSPECTION REPORT
# ============================================================

report_lines = []

report_lines.append("# Phase 6D: Manual Error Inspection Report\n")
report_lines.append(
    "This report summarizes model errors identified during "
    "Phase 6B and Phase 6C.\n"
)

report_lines.append("## 1. Input Files\n")

report_lines.append(
    f"- Phase 6B directory: `{PHASE6B_DIR}`"
)

report_lines.append(
    f"- Phase 6C directory: `{PHASE6C_DIR}`"
)

report_lines.append("\n## 2. Interpretation Guidelines\n")

report_lines.append(
    "- False positives are unsupported claims predicted as supported."
)

report_lines.append(
    "- False negatives are supported claims predicted as unsupported."
)

report_lines.append(
    "- High-confidence errors indicate potentially overconfident "
    "model predictions."
)

report_lines.append(
    "- Qualitative error patterns are heuristic associations and "
    "should not be interpreted as causal explanations."
)

if source_error_summary is not None:

    report_lines.append("\n## 3. Source-wise Error Summary\n")

    report_lines.append(
        source_error_summary.to_markdown(index=False)
    )

if pattern_data is not None:

    report_lines.append("\n## 4. Qualitative Error Patterns\n")

    report_lines.append(
        pattern_data.to_markdown(index=False)
    )

if text_statistics is not None:

    report_lines.append("\n## 5. Error Text Statistics\n")

    report_lines.append(
        text_statistics.to_markdown(index=False)
    )

if high_confidence_errors is not None:

    report_lines.append(
        "\n## 6. High-Confidence Error Examples\n"
    )

    report_lines.append(
        f"Number of high-confidence errors: "
        f"{len(high_confidence_errors)}"
    )

    columns_to_display = [
        column
        for column in [
            "sample_id",
            "source_dataset",
            "claim_text",
            "evidence_text",
            "true_label",
            "predicted_label",
            "probability",
            "error_type",
            "error_pattern"
        ]
        if column in high_confidence_errors.columns
    ]

    if len(columns_to_display) > 0:

        examples = high_confidence_errors[
            columns_to_display
        ].head(20)

        report_lines.append(
            examples.to_markdown(index=False)
        )

report_lines.append("\n## 7. Suggested Future Improvements\n")

report_lines.append(
    "1. Manually inspect false positives involving uncertainty "
    "or hedging language."
)

report_lines.append(
    "2. Analyze whether evidence length affects classification."
)

report_lines.append(
    "3. Investigate source-specific calibration differences."
)

report_lines.append(
    "4. Consider confidence calibration techniques."
)

report_lines.append(
    "5. Evaluate additional linguistic and semantic features."
)

report_lines.append(
    "6. Test a separate model or threshold for different sources "
    "only if supported by validation experiments."
)

report_path = OUTPUT_DIR / "phase6d_manual_inspection_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

print(f"Saved report: {report_path}")


# ============================================================
# 12. SAVE SUMMARY JSON
# ============================================================

summary = {
    "phase": "6D",
    "phase_name": "Manual Inspection and Visualization",
    "input_directories": {
        "phase6b": str(PHASE6B_DIR),
        "phase6c": str(PHASE6C_DIR)
    },
    "output_directory": str(OUTPUT_DIR),
    "files_generated": [
        file.name
        for file in OUTPUT_DIR.iterdir()
        if file.is_file()
    ],
    "notes": [
        "Qualitative error patterns are heuristic.",
        "High-confidence errors should be manually reviewed.",
        "Source-wise conclusions should consider sample size."
    ]
}

summary_path = OUTPUT_DIR / "phase6d_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print(f"Saved summary: {summary_path}")


# ============================================================
# 13. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6D COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nAll outputs saved in:\n{OUTPUT_DIR}")

print("\nReview the following files:")
print("1. error_pattern_distribution.png")
print("2. source_error_comparison.png")
print("3. error_type_by_source.png")
print("4. confidence_error_distribution.png")
print("5. phase6d_manual_inspection_report.md")
print("6. phase6d_summary.json")