from datasets import load_dataset
import os
import json

DATASET_NAME = "zzha6204/MM-Health"

OUTPUT_DIR = (
    "outputs/phase9/mm_health_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSETS = [
    "Med-MMHL",
    "MM-COVID19",
    "ReCOVery",
    "MMCoVar"
]

print("Loading MM-Health dataset...")

dataset = load_dataset(
    DATASET_NAME,
    streaming=True
)

results = []

for split_name, split_data in dataset.items():

    print("\nProcessing split:", split_name)

    outer_record = next(iter(split_data))

    for subset_name in SUBSETS:

        subset_records = outer_record.get(
            subset_name,
            []
        )

        if not isinstance(subset_records, list):
            continue

        print(
            "\nSubset:",
            subset_name
        )

        for item in subset_records[:3]:

            image_data = item.get(
                "image",
                {}
            )

            text_data = item.get(
                "text",
                {}
            )

            print("\nRecord ID:", item.get("id"))

            print(
                "Label:",
                item.get("label")
            )

            print(
                "Source:",
                item.get("source")
            )

            print(
                "English:",
                item.get("is_english")
            )

            print(
                "Image type:",
                type(image_data)
            )

            if isinstance(image_data, dict):

                print(
                    "Image keys:",
                    list(image_data.keys())
                )

                print(
                    "Original image:",
                    image_data.get("original")
                )

            print(
                "Text type:",
                type(text_data)
            )

            if isinstance(text_data, dict):

                text_keys = list(
                    text_data.keys()
                )

                print(
                    "Text keys:",
                    text_keys
                )

                for key in text_keys[:2]:

                    text_value = text_data.get(
                        key
                    )

                    print(
                        "Text sample:",
                        str(text_value)[:300]
                    )

            results.append({
                "split": split_name,
                "subset": subset_name,
                "id": item.get("id"),
                "label": item.get("label"),
                "source": item.get("source"),
                "is_english": item.get("is_english"),
                "image_keys": (
                    list(image_data.keys())
                    if isinstance(
                        image_data,
                        dict
                    )
                    else []
                ),
                "text_keys": (
                    list(text_data.keys())
                    if isinstance(
                        text_data,
                        dict
                    )
                    else []
                )
            })

output_path = os.path.join(
    OUTPUT_DIR,
    "mm_health_content_verification.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        default=str
    )

print("\nVerification saved to:")
print(output_path)

print(
    "\nMM-Health content verification completed."
)