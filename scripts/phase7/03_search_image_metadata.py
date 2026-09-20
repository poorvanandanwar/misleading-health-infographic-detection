
from pathlib import Path
import json
import csv
import re
from collections import Counter

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "phase7"
    / "phase7c_metadata_search"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keywords used to identify possible metadata files
IMAGE_KEYWORDS = [
    "image",
    "img",
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "image_path",
    "image_url",
]

LABEL_KEYWORDS = [
    "label",
    "target",
    "class",
    "category",
    "verdict",
    "truth",
    "annotation",
]

HEALTH_KEYWORDS = [
    "health",
    "medical",
    "misleading",
    "infographic",
    "claim",
    "evidence",
    "fact",
]

SUPPORTED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".txt",
    ".tsv",
}

results = []
extension_counts = Counter()

print("=" * 70)
print("PHASE 7C - IMAGE METADATA SEARCH")
print("=" * 70)
print(f"Searching directory: {DATA_DIR}")
print()

def flatten_keys(obj, prefix=""):
    """Extract nested dictionary keys."""
    keys = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full_key)
            keys.extend(flatten_keys(value, full_key))

    elif isinstance(obj, list):
        for item in obj[:5]:
            keys.extend(flatten_keys(item, prefix))

    return keys


def keyword_matches(keys, keywords):
    joined_keys = " ".join(keys).lower()
    return [
        keyword for keyword in keywords
        if keyword in joined_keys
    ]


def inspect_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        keys = flatten_keys(data)

        image_matches = keyword_matches(keys, IMAGE_KEYWORDS)
        label_matches = keyword_matches(keys, LABEL_KEYWORDS)
        health_matches = keyword_matches(keys, HEALTH_KEYWORDS)

        return {
            "image_matches": image_matches,
            "label_matches": label_matches,
            "health_matches": health_matches,
            "keys": keys[:100],
            "status": "valid",
        }

    except Exception as error:
        return {
            "image_matches": [],
            "label_matches": [],
            "health_matches": [],
            "keys": [],
            "status": f"error: {str(error)[:150]}",
        }


def inspect_jsonl_file(file_path):
    keys = []
    line_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    keys.extend(flatten_keys(data))
                    line_count += 1

                if line_count >= 10:
                    break

        image_matches = keyword_matches(keys, IMAGE_KEYWORDS)
        label_matches = keyword_matches(keys, LABEL_KEYWORDS)
        health_matches = keyword_matches(keys, HEALTH_KEYWORDS)

        return {
            "image_matches": image_matches,
            "label_matches": label_matches,
            "health_matches": health_matches,
            "keys": list(dict.fromkeys(keys))[:100],
            "status": "valid",
        }

    except Exception as error:
        return {
            "image_matches": [],
            "label_matches": [],
            "health_matches": [],
            "keys": [],
            "status": f"error: {str(error)[:150]}",
        }


def inspect_text_file(file_path):
    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:
            text = file.read(50000).lower()

        image_matches = [
            keyword for keyword in IMAGE_KEYWORDS
            if keyword in text
        ]

        label_matches = [
            keyword for keyword in LABEL_KEYWORDS
            if keyword in text
        ]

        health_matches = [
            keyword for keyword in HEALTH_KEYWORDS
            if keyword in text
        ]

        return {
            "image_matches": image_matches,
            "label_matches": label_matches,
            "health_matches": health_matches,
            "keys": [],
            "status": "valid",
        }

    except Exception as error:
        return {
            "image_matches": [],
            "label_matches": [],
            "health_matches": [],
            "keys": [],
            "status": f"error: {str(error)[:150]}",
        }


# Search all metadata files
all_files = [
    file_path
    for file_path in DATA_DIR.rglob("*")
    if file_path.is_file()
    and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
]

print(f"Metadata files found: {len(all_files)}")
print()

for index, file_path in enumerate(all_files, start=1):
    extension = file_path.suffix.lower()
    extension_counts[extension] += 1

    relative_path = file_path.relative_to(BASE_DIR)

    if extension == ".json":
        inspection = inspect_json_file(file_path)

    elif extension == ".jsonl":
        inspection = inspect_jsonl_file(file_path)

    else:
        inspection = inspect_text_file(file_path)

    image_matches = inspection["image_matches"]
    label_matches = inspection["label_matches"]
    health_matches = inspection["health_matches"]

    relevance_score = (
        len(image_matches)
        + len(label_matches)
        + len(health_matches)
    )

    if relevance_score > 0:
        result = {
            "file_path": str(relative_path),
            "extension": extension,
            "image_keywords": ", ".join(image_matches),
            "label_keywords": ", ".join(label_matches),
            "health_keywords": ", ".join(health_matches),
            "relevance_score": relevance_score,
            "status": inspection["status"],
            "sample_keys": " | ".join(inspection["keys"][:30]),
        }

        results.append(result)

        print(f"[{len(results)}] Candidate file:")
        print(f"    {relative_path}")
        print(f"    Image keywords: {image_matches}")
        print(f"    Label keywords: {label_matches}")
        print(f"    Health keywords: {health_matches}")
        print()

# Save results
results.sort(
    key=lambda item: item["relevance_score"],
    reverse=True
)

output_csv = OUTPUT_DIR / "image_metadata_candidates.csv"

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    fieldnames = [
        "file_path",
        "extension",
        "image_keywords",
        "label_keywords",
        "health_keywords",
        "relevance_score",
        "status",
        "sample_keys",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

summary = {
    "metadata_files_found": len(all_files),
    "candidate_files_found": len(results),
    "extension_counts": dict(extension_counts),
    "files_with_image_keywords": sum(
        bool(item["image_keywords"])
        for item in results
    ),
    "files_with_label_keywords": sum(
        bool(item["label_keywords"])
        for item in results
    ),
    "files_with_both_image_and_label_keywords": sum(
        bool(item["image_keywords"])
        and bool(item["label_keywords"])
        for item in results
    ),
}

summary_path = OUTPUT_DIR / "phase7c_summary.json"

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
print("PHASE 7C SUMMARY")
print("=" * 70)
print(f"Metadata files found: {summary['metadata_files_found']}")
print(f"Candidate files found: {summary['candidate_files_found']}")
print(
    "Files with image keywords:",
    summary["files_with_image_keywords"]
)
print(
    "Files with label keywords:",
    summary["files_with_label_keywords"]
)
print(
    "Files with both image and label keywords:",
    summary["files_with_both_image_and_label_keywords"]
)
print()
print(f"Saved CSV: {output_csv}")
print(f"Saved summary: {summary_path}")
print()
print("Phase 7C completed.")
