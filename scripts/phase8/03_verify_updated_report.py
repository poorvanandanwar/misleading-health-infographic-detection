from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    BASE_DIR
    / "outputs"
    / "phase8"
    / "phase8b_updated_research_report"
    / "updated_research_findings_report.md"
)

print("=" * 70)
print("PHASE 8C - VERIFY UPDATED RESEARCH REPORT")
print("=" * 70)

if not REPORT_PATH.exists():
    print("ERROR: Updated report not found.")
    print(REPORT_PATH)
    raise SystemExit(1)

with open(REPORT_PATH, "r", encoding="utf-8") as file:
    report = file.read()

required_sections = [
    "## 8. Phase 7: Image Dataset Investigation",
    "### 8.1 Image Dataset Preparation",
    "### 8.2 Image-Label Alignment",
    "### 8.4 Image Modeling Decision",
    "## 9. Updated Overall Conclusion",
]

print(f"Report found: {REPORT_PATH}")
print(f"Total characters: {len(report)}")
print(f"Total lines: {len(report.splitlines())}")

print("\nSECTION VERIFICATION")

all_present = True

for section in required_sections:
    if section in report:
        print(f"[PASS] {section}")
    else:
        print(f"[FAIL] {section}")
        all_present = False

print("\nOUTDATED STATEMENT CHECK")

if "The qualitative pattern summary was unavailable." not in report:
    print("[PASS] Outdated qualitative statement removed")
else:
    print("[FAIL] Outdated qualitative statement still exists")
    all_present = False

print("\nFINAL RESULT")

if all_present:
    print("Phase 8C verification completed successfully")
else:
    print("Phase 8C verification failed")
    raise SystemExit(1)