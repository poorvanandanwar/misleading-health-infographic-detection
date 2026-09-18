
from pathlib import Path
import csv
import json
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase7"
    / "phase7e_health_image_search"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

HEALTH_KEYWORDS = [
    "health",
    "medical",
    "medicine",
    "disease",
    "clinical",
    "infographic",
    "misleading",
    "misinformation",
    "fact",
    "claim",
    "pubhealth",
    "healthfc"
]

print("=" * 70)
print("PHASE 7E - SEARCH FOR HEALTH-RELATED IMAGES")
print("=" * 70)

all_images = [
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images found: {len(all_images)}")

results = []

for image_path in all_images:
    path_text = str(image_path.relative_to(BASE_DIR)).lower()

    matched_keywords = [
        keyword
        for keyword in HEALTH_KEYWORDS
        if keyword in path_text
    ]

    if matched_keywords:
        results.append({
            "image_path": str(image_path.relative_to(BASE_DIR)),
            "matched_keywords": ", ".join(matched_keywords)
        })

print(f"Images in health-related paths: {len(results)}")

for item in results[:30]:
    print(item["image_path"])
    print("Keywords:", item["matched_keywords"])
    print()

output_csv = OUTPUT_DIR / "health_image_candidates.csv"

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path",
            "matched_keywords"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

summary = {
    "total_images_found": len(all_images),
    "health_path_candidates": len(results),
    "note": (
        "Filename and directory keyword matching is heuristic. "
        "It does not prove that images contain health content."
    )
}

summary_path = OUTPUT_DIR / "phase7e_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )

print("=" * 70)
print("PHASE 7E COMPLETED")
print("=" * 70)
print(f"Total images: {len(all_images)}")
print(f"Health path candidates: {len(results)}")
print(f"Saved CSV: {output_csv}")
print(f"Saved summary: {summary_path}")
