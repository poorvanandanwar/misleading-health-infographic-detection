
from pathlib import Path
import json
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase7"
    / "phase7f_final_image_dataset_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


audit = {
    "phase": "Phase 7F - Final Image Dataset Audit",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_images_found": 20882,
    "valid_images": 20882,
    "invalid_images": 0,
    "health_path_candidates": 3,
    "verified_health_images": 0,
    "image_label_mapping_available": False,
    "misleading_health_labels_available": False,
    "dataset_source": "ChartQA",
    "dataset_suitable_for_supervised_health_misinformation_training": False,
    "conclusion": (
        "The available images belong to the ChartQA dataset. "
        "Although three filenames contain health-related keywords, "
        "there is no verified image-to-health-misinformation label mapping. "
        "The dataset should not be used for supervised training of a "
        "misleading health infographic classifier."
    ),
    "recommended_action": (
        "Document the image dataset limitation and focus on the "
        "text-based misinformation detection pipeline unless a suitable "
        "labeled health infographic dataset is obtained."
    )
}


summary_path = OUTPUT_DIR / "phase7f_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=4)


report_path = OUTPUT_DIR / "phase7f_final_image_dataset_audit.md"

report = f"""# Phase 7F - Final Image Dataset Audit

## Dataset Summary

- Total images found: {audit["total_images_found"]}
- Valid images: {audit["valid_images"]}
- Invalid images: {audit["invalid_images"]}
- Health path candidates: {audit["health_path_candidates"]}
- Verified health images: {audit["verified_health_images"]}

## Label Availability

- Image-label mapping available: No
- Misleading health labels available: No
- Dataset source: ChartQA

## Conclusion

{audit["conclusion"]}

## Recommended Action

{audit["recommended_action"]}
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)


print("=" * 70)
print("PHASE 7F - FINAL IMAGE DATASET AUDIT")
print("=" * 70)
print(f"Summary saved: {summary_path}")
print(f"Report saved: {report_path}")
print("Image dataset suitable for supervised training: NO")
print("Phase 7F completed successfully")
