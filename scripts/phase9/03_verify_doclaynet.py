import json
import os

INPUT_PATH = (
    "outputs/phase9/doclaynet_inspection/"
    "doclaynet_v12_structure.json"
)

print("Verifying DocLayNet inspection output...")

if not os.path.exists(INPUT_PATH):
    print("\nERROR: Structure file not found:")
    print(INPUT_PATH)
    raise SystemExit

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

print("\nAvailable splits:")
print(list(data.keys()))

for split_name, split_info in data.items():

    print("\n" + "=" * 60)
    print("Split:", split_name)

    columns = split_info.get("columns", [])

    print("Columns:")
    for column in columns:
        print(" -", column)

    image_present = "image" in columns

    annotation_keywords = [
        "bbox",
        "bounding_box",
        "category",
        "label",
        "segmentation"
    ]

    annotation_columns = [
        column
        for column in columns
        if any(
            keyword in column.lower()
            for keyword in annotation_keywords
        )
    ]

    print("\nImage column present:", image_present)

    print("Possible annotation columns:")
    print(annotation_columns)

print("\nDocLayNet verification completed.")