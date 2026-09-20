from datasets import load_dataset
import os
import json

DATASET_NAME = "docling-project/DocLayNet-v1.2"

OUTPUT_DIR = "outputs/phase9/doclaynet_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading DocLayNet v1.2 in streaming mode...")
print("Dataset:", DATASET_NAME)

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
        print("Split:", split_name)

        sample = next(iter(split_data))

        print("\nColumns:")
        print(list(sample.keys()))

        print("\nSample information:")

        for key, value in sample.items():

            if key == "image":
                print(key, ":", type(value))

            else:
                value_text = str(value)

                if len(value_text) > 500:
                    value_text = value_text[:500] + "..."

                print(key, ":", value_text)

        summary[split_name] = {
            "columns": list(sample.keys())
        }

    output_path = os.path.join(
        OUTPUT_DIR,
        "doclaynet_v12_structure.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    print("\nSummary saved to:")
    print(output_path)

    print("\nDocLayNet v1.2 inspection completed.")

except Exception as error:

    print("\nInspection failed:")
    print(type(error).__name__, error)