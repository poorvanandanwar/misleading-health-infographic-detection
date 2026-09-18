import pandas as pd
import os
import json

BASE_DIR = (
    "outputs/phase9/mm_health_analysis/"
    "clean_dataset"
)

FILES = {
    "train": "mm_health_train_clean.csv",
    "validation": "mm_health_validation_clean.csv",
    "test": "mm_health_test_clean.csv"
}

print("Loading clean MM-Health splits...")

datasets = {}

for split_name, file_name in FILES.items():

    file_path = os.path.join(
        BASE_DIR,
        file_name
    )

    df = pd.read_csv(file_path)

    datasets[split_name] = df

    print(
        f"\n{split_name.capitalize()} shape:",
        df.shape
    )

    print("Label distribution:")

    print(
        df["label"].value_counts(
            dropna=False
        )
    )

# Extract record keys
train_keys = set(
    datasets["train"]["record_key"]
)

validation_keys = set(
    datasets["validation"]["record_key"]
)

test_keys = set(
    datasets["test"]["record_key"]
)

# Check overlaps
train_validation = train_keys.intersection(
    validation_keys
)

train_test = train_keys.intersection(
    test_keys
)

validation_test = validation_keys.intersection(
    test_keys
)

print("\n" + "=" * 60)
print("OVERLAP VERIFICATION")

print(
    "Train-validation overlap:",
    len(train_validation)
)

print(
    "Train-test overlap:",
    len(train_test)
)

print(
    "Validation-test overlap:",
    len(validation_test)
)

# Check missing values
print("\n" + "=" * 60)
print("MISSING VALUE CHECK")

for split_name, df in datasets.items():

    print("\n", split_name)

    print(
        df[
            [
                "record_key",
                "label",
                "original_image"
            ]
        ].isnull().sum()
    )

# Save verification summary
summary = {
    "train_records": len(train_keys),
    "validation_records": len(validation_keys),
    "test_records": len(test_keys),
    "train_validation_overlap": len(
        train_validation
    ),
    "train_test_overlap": len(
        train_test
    ),
    "validation_test_overlap": len(
        validation_test
    ),
    "status": (
        "PASS"
        if (
            len(train_validation) == 0
            and len(train_test) == 0
            and len(validation_test) == 0
        )
        else "FAIL"
    )
}

summary_path = os.path.join(
    BASE_DIR,
    "clean_split_verification_summary.json"
)

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

print("\nSummary saved to:")
print(summary_path)

if summary["status"] == "PASS":

    print(
        "\nPASS: No overlap between clean splits."
    )

else:

    print(
        "\nFAIL: Overlap detected."
    )

print("\nClean split verification completed.")