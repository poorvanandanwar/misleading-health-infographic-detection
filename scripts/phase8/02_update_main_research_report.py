from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_REPORT = (
    BASE_DIR
    / "outputs"
    / "phase6"
    / "phase6e_research_findings"
    / "phase6e_research_findings_report.md"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase8"
    / "phase8b_updated_research_report"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT = OUTPUT_DIR / "updated_research_findings_report.md"


if not INPUT_REPORT.exists():
    print("ERROR: Original research report not found.")
    print(INPUT_REPORT)
    raise SystemExit(1)


with open(INPUT_REPORT, "r", encoding="utf-8") as file:
    original_report = file.read()


# Correct the outdated statement regarding qualitative analysis
original_report = original_report.replace(
    "The qualitative pattern summary was unavailable.",
    (
        "Qualitative analysis identified recurring patterns in incorrect "
        "predictions, including uncertainty or hedging language, "
        "numerical/statistical claims, causal relationship language, "
        "strong or exaggerated claims, and comparative claims."
    )
)


phase7_section = f"""

## 8. Phase 7: Image Dataset Investigation

### 8.1 Image Dataset Preparation

The image dataset investigation identified 20,882 images.

All 20,882 images were successfully validated as readable images.
No invalid images were identified.

The images were located in the ChartQA dataset directory.

### 8.2 Image-Label Alignment

The PubHealth dataset contains textual claims, explanations,
and labels but does not contain image-related columns.

The ChartQA dataset contains image, question, and answer mappings.
However, it does not provide labels indicating whether health
information is misleading or reliable.

Therefore, no verified image-to-misleading-health-information
label mapping was identified.

### 8.3 Health-Related Image Search

A keyword search identified three possible health-related image paths.

Two candidates included terms associated with health spending
and medical graduates.

One candidate was a false positive caused by the keyword
"fact" appearing in a filename.

Filename-based keyword matching cannot confirm that an image
is a health infographic.

### 8.4 Image Modeling Decision

A supervised image classification model was not trained.

The available images do not contain verified labels for
misleading or reliable health information. Training a supervised
model without appropriate target labels would not provide
scientifically reliable results.

### 8.5 Image Dataset Limitation

The currently available image data belongs to the general
ChartQA dataset rather than a dedicated health misinformation
dataset.

The project therefore focuses on the text-based misinformation
detection pipeline using the available PubHealth and HealthFC data.

Image-based and multimodal detection remain possible areas
for future work if a suitable labeled health infographic dataset
is obtained.

### 8.6 Future Research Direction

Future work may include:

- Obtaining a labeled health infographic dataset.
- Creating expert-annotated labels for health information.
- Combining image and text representations.
- Investigating multimodal fusion architectures.
- Evaluating visual explanations alongside textual explanations.

## 9. Updated Overall Conclusion

The text-based model demonstrates useful classification performance,
but its errors vary across data sources and linguistic patterns.

The image dataset investigation did not identify a suitable
labeled dataset for supervised health misinformation image
classification.

Consequently, image-based modeling was not performed in the
current study.

The findings highlight the importance of reliable labels,
cross-source evaluation, confidence analysis, and future
multimodal research.

Report updated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


updated_report = original_report.rstrip() + phase7_section


with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    file.write(updated_report)


print("=" * 70)
print("PHASE 8B - UPDATED RESEARCH REPORT")
print("=" * 70)
print(f"Original report: {INPUT_REPORT}")
print(f"Updated report: {OUTPUT_REPORT}")
print(f"Total characters: {len(updated_report)}")
print(f"Total lines: {len(updated_report.splitlines())}")
print("Original report preserved: YES")
print("Phase 8B completed successfully")