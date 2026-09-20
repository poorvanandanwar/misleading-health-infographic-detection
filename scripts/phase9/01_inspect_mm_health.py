from datasets import load_dataset
import json
import os

DATASET_NAME = "zzha6204/MM-Health"

OUTPUT_DIR = "outputs/phase9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MM-Health dataset...")
print("Streaming mode enabled.")

try:
    dataset = load_dataset(
        DATASET_NAME,
        streaming=True
    )

    print("\nDataset structure:")
    print(dataset)

    summary = {}

    for split_name, split_data in dataset.items():

        print("\n" + "=" * 60)
        print(f"Split: {split_name}")

        sample = next(iter(split_data))

        print("\nColumns:")
        print(list(sample.keys()))

        print("\nSample information:")

        for key, value in sample.items():

            if key == "image":
                print(f"{key}: {type(value)}")

            else:
                value_text = str(value)

                if len(value_text) > 500:
                    value_text = value_text[:500] + "..."

                print(f"{key}: {value_text}")

        summary[split_name] = {
            "columns": list(sample.keys())
        }

    output_path = os.path.join(
        OUTPUT_DIR,
        "mm_health_dataset_structure.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nSummary saved to:")
    print(output_path)

except Exception as error:

    print("\nDataset loading failed:")
    print(type(error).__name__, error)