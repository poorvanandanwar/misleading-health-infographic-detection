import os
from datetime import datetime

OUTPUT_DIR = "outputs/phase9/doclaynet_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "doclaynet_suitability_report.md"
)

report = f"""# DocLayNet Suitability Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Dataset

- Dataset: DocLayNet v1.2
- Purpose: Document layout analysis
- Modalities: Layout annotations and page images

## 2. Verified Features

The dataset contains:

- Page images
- Bounding boxes
- Category IDs
- Segmentation coordinates
- Area information
- PDF cell information
- Document metadata
- Layout modality information

## 3. Relevance to the Project

The project focuses on detecting misleading health information
using image and text features.

DocLayNet provides document-layout annotations, but it does not
provide verified health misinformation labels.

The inspected sample belonged to a financial report document.

## 4. Suitability Decision

DocLayNet is NOT suitable as the primary supervised dataset
for health misinformation classification.

It MAY be used as an auxiliary dataset for:

- Document layout understanding
- Region detection
- Text and visual region separation
- Layout-aware preprocessing

## 5. Limitations

- No verified misinformation labels
- No health-specific classification labels
- Domain mismatch with health infographics
- Layout categories do not indicate whether content is true or false

## 6. Final Decision

DocLayNet will not be used directly to train the primary
health misinformation classifier.

It may be considered for future auxiliary layout experiments.
"""

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(report)

print("Report saved to:")
print(REPORT_PATH)

print("\nDocLayNet suitability report completed.")