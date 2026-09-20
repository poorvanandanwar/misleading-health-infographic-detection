from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "phase8"
    / "phase8d_final_project_summary"
    / "final_project_summary.md"
)

print("=" * 70)
print("PHASE 8E - VERIFY FINAL PROJECT SUMMARY")
print("=" * 70)

if not SUMMARY_PATH.exists():
    print("ERROR: Final project summary not found.")
    print(SUMMARY_PATH)
    raise SystemExit(1)

with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
    summary = file.read()

required_sections = [
    "# Final Project Summary",
    "## 1. Project Objective",
    "## 2. Datasets",
    "## 3. Text-Based Modeling",
    "## 4. Overall Text Model Performance",
    "## 5. Source-Wise Evaluation",
    "## 6. Error Analysis",
    "## 7. Image Dataset Investigation",
    "## 8. Main Limitations",
    "## 9. Future Work",
    "## 10. Conclusion",
]

print(f"Summary found: {SUMMARY_PATH}")
print(f"Total characters: {len(summary)}")
print(f"Total lines: {len(summary.splitlines())}")

print("\nSECTION VERIFICATION")

all_present = True

for section in required_sections:
    if section in summary:
        print(f"[PASS] {section}")
    else:
        print(f"[FAIL] {section}")
        all_present = False

print("\nMETRIC VERIFICATION")

required_metrics = [
    "Accuracy: 84.33%",
    "Precision: 75.96%",
    "Recall: 87.93%",
    "F1-score: 81.51%",
    "ROC-AUC: 92.08%",
]

for metric in required_metrics:
    if metric in summary:
        print(f"[PASS] {metric}")
    else:
        print(f"[FAIL] {metric}")
        all_present = False

print("\nFINAL RESULT")

if all_present:
    print("Phase 8E verification completed successfully")
else:
    print("Phase 8E verification failed")
    raise SystemExit(1)