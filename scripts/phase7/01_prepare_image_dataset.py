
from pathlib import Path
import pandas as pd
from PIL import Image
import json

# Project directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase7" / "phase7a_dataset_preparation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 70)
print("PHASE 7A: IMAGE DATASET INSPECTION")
print("=" * 70)

# Find all images
image_files = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"\nTotal images found: {len(image_files)}")

if not image_files:
    raise FileNotFoundError("No image files found inside the data directory.")

# Inspect image properties
records = []

for image_path in image_files:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode

        records.append({
            "image_path": str(image_path.relative_to(BASE_DIR)),
            "filename": image_path.name,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode,
            "valid": True,
            "error": ""
        })

    except Exception as error:
        records.append({
            "image_path": str(image_path.relative_to(BASE_DIR)),
            "filename": image_path.name,
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "valid": False,
            "error": str(error)
        })

image_df = pd.DataFrame(records)

# Save image inventory
inventory_path = OUTPUT_DIR / "image_inventory.csv"
image_df.to_csv(inventory_path, index=False)

# Summary
valid_images = int(image_df["valid"].sum())
invalid_images = int((~image_df["valid"]).sum())

summary = {
    "total_images": len(image_df),
    "valid_images": valid_images,
    "invalid_images": invalid_images,
    "unique_formats": image_df["format"].dropna().unique().tolist(),
    "width_statistics": image_df["width"].describe().to_dict(),
    "height_statistics": image_df["height"].describe().to_dict()
}

with open(OUTPUT_DIR / "phase7a_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, default=str)

print("\nImage validation completed.")
print(f"Valid images: {valid_images}")
print(f"Invalid images: {invalid_images}")

print("\nImage formats:")
print(image_df["format"].value_counts(dropna=False))

print("\nSample image records:")
print(image_df.head(10).to_string(index=False))

print(f"\nInventory saved to: {inventory_path}")
print(f"Summary saved to: {OUTPUT_DIR / 'phase7a_summary.json'}")

print("\nPHASE 7A COMPLETED SUCCESSFULLY")
