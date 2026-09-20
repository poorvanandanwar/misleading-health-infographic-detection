from datasets import load_dataset
from PIL import Image
import os
import json

DATASET_NAME = "zzha6204/MM-Health"

OUTPUT_DIR = (
    "outputs/phase9/mm_health_analysis/"
    "image_loading_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSETS = [
    "Med-MMHL",
    "MM-COVID19",
    "ReCOVery",
    "MMCoVar"
]

results = []

print("Loading MM-Health dataset...")

dataset = load_dataset(
    DATASET_NAME,
    streaming=True
)

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

        for item in subset_records[:3]:

            record_id = str(item.get("id"))
            image_data = item.get("image", {})

            original_images = image_data.get(
                "original",
                []
            )

            if isinstance(original_images, str):
                original_images = [original_images]

            print("\nSubset:", subset_name)
            print("Record ID:", record_id)
            print("Image references:", original_images)

            for image_reference in original_images[:1]:

                result = {
                    "split": split_name,
                    "subset": subset_name,
                    "record_id": record_id,
                    "image_reference": str(
                        image_reference
                    ),
                    "reference_type": type(
                        image_reference
                    ).__name__,
                    "accessible": False,
                    "error": ""
                }

                try:

                    if isinstance(
                        image_reference,
                        Image.Image
                    ):

                        image = image_reference

                        result["accessible"] = True
                        result["image_size"] = image.size
                        result["image_mode"] = image.mode

                    elif isinstance(
                        image_reference,
                        str
                    ):

                        if os.path.exists(
                            image_reference
                        ):

                            image = Image.open(
                                image_reference
                            )

                            result["accessible"] = True
                            result["image_size"] = image.size
                            result["image_mode"] = image.mode

                        else:

                            result["error"] = (
                                "Image path is not "
                                "available locally"
                            )

                    else:

                        result["error"] = (
                            "Unsupported image type"
                        )

                except Exception as error:

                    result["error"] = str(error)

                print(
                    "Accessible:",
                    result["accessible"]
                )

                print(
                    "Error:",
                    result["error"]
                )

                results.append(result)

output_path = os.path.join(
    OUTPUT_DIR,
    "image_loading_test.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

print("\nImage loading test completed.")
print("Saved to:", output_path)