from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase8"
    / "phase8f_deliverables_index"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUTPUT_DIR / "project_deliverables_index.md"


important_paths = {
    "Phase 5": BASE_DIR / "outputs" / "phase5",
    "Phase 6": BASE_DIR / "outputs" / "phase6",
    "Phase 7": BASE_DIR / "outputs" / "phase7",
    "Phase 8": BASE_DIR / "outputs" / "phase8",
    "Scripts": BASE_DIR / "scripts",
    "Processed Data": BASE_DIR / "data" / "processed",
}


lines = [
    "# Project Deliverables Index",
    "",
    "## Project",
    "",
    "Misleading Health Information Detection Using Data Science Techniques",
    "",
    "## Available Project Directories",
    "",
]

for name, path in important_paths.items():
    status = "AVAILABLE" if path.exists() else "NOT FOUND"
    lines.append(f"- **{name}:** `{path}` — {status}")

lines.extend([
    "",
    "## Important Reports",
    "",
    "- Updated research findings report",
    "- Final project summary",
    "- Phase 6 research findings",
    "- Phase 7 image limitation report",
    "",
    "## Main Model Results",
    "",
    "- Accuracy: 84.33%",
    "- Precision: 75.96%",
    "- Recall: 87.93%",
    "- F1-score: 81.51%",
    "- ROC-AUC: 92.08%",
    "",
    "## Image Dataset Decision",
    "",
    "Supervised image classification was not performed because",
    "a verified image-to-misleading-health-information label",
    "mapping was not identified.",
    "",
    "## Project Status",
    "",
    "Text-based misinformation detection pipeline completed.",
    "Image dataset investigation completed.",
    "Research report and final summary generated.",
    "",
    f"Index generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
])


with open(INDEX_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("=" * 70)
print("PHASE 8F - PROJECT DELIVERABLES INDEX")
print("=" * 70)
print(f"Index saved to: {INDEX_PATH}")
print(f"Total lines: {len(lines)}")
print("Phase 8F completed successfully")