from datasets import load_dataset
import os
import json

DATASET_NAME = "docling-project/DocLayNet-v1.2"

OUTPUT_DIR = "outputs/phase9/doclaynet_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading DocLayNet annotations...")

try:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    sample = next(iter(dataset))

    print("\nSample keys:")
    print(list(sample.keys()))

    print("\nAnnotation details:")

    for key in [
        "bboxes",
        "category_id",
        "segmentation",
        "area",
        "metadata",
        "modalities"
    ]:

        if key in sample:

            value = sample[key]

            print("\nField:", key)
            print("Type:", type(value))

            value_text = str(value)

            if len(value_text) > 1000:
                value_text = value_text[:1000] + "..."

            print("Value:", value_text)

    output_path = os.path.join(
        OUTPUT_DIR,
        "doclaynet_annotation_sample.json"
    )

    safe_sample = {
        key: str(value)
        for key, value in sample.items()
        if key != "image"
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(safe_sample, file, indent=4)

    print("\nAnnotation sample saved to:")
    print(output_path)

    print("\nPhase 9 annotation inspection completed.")

except Exception as error:

    print("\nInspection failed:")
    print(type(error).__name__, error)