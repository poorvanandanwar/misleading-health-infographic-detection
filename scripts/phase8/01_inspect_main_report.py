from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    BASE_DIR
    / "outputs"
    / "phase6"
    / "phase6e_research_findings"
    / "phase6e_research_findings_report.md"
)


print("=" * 70)
print("PHASE 8A - INSPECT MAIN RESEARCH REPORT")
print("=" * 70)


if not REPORT_PATH.exists():
    print("ERROR: Main research report not found.")
    print(f"Expected path: {REPORT_PATH}")
else:
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        content = file.read()

    print(f"Report found: {REPORT_PATH}")
    print(f"Total characters: {len(content)}")
    print(f"Total lines: {len(content.splitlines())}")

    print("\n" + "=" * 70)
    print("EXISTING REPORT CONTENT")
    print("=" * 70)

    print(content)

    print("\n" + "=" * 70)
    print("PHASE 8A COMPLETED")
    print("=" * 70)