from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "chartqa"
    / "chartqa_multimodal_pilot.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "phase4"
    / "health_chart_candidates.csv"
)


# ============================================================
# KEYWORDS
# ============================================================

HEALTH_KEYWORDS = [
    "health",
    "medical",
    "medicine",
    "disease",
    "diseases",
    "cancer",
    "diabetes",
    "diabetic",
    "obesity",
    "vaccine",
    "vaccination",
    "virus",
    "viral",
    "covid",
    "coronavirus",
    "hospital",
    "doctor",
    "patient",
    "patients",
    "medication",
    "treatment",
    "symptom",
    "symptoms",
    "mortality",
    "death",
    "deaths",
    "life expectancy",
    "smoking",
    "smoker",
    "tobacco",
    "alcohol",
    "nutrition",
    "diet",
    "pregnancy",
    "maternal",
    "mental health",
    "depression",
    "anxiety",
    "blood",
    "heart",
    "cardiovascular",
    "stroke",
    "hiv",
    "aids",
    "malaria",
    "tuberculosis",
    "public health",
    "healthcare",
    "health care"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4D - HEALTH-RELATED CHART CANDIDATE FILTER")
    print("=" * 70)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    # --------------------------------------------------------
    # Search OCR + question
    # --------------------------------------------------------

    searchable = (
        df[
            [
                "ocr_text",
                "question"
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(
            " ".join,
            axis=1
        )
        .str.lower()
    )

    keyword_matches = []

    for text in searchable:

        found = []

        for keyword in HEALTH_KEYWORDS:

            if keyword in text:

                found.append(
                    keyword
                )

        keyword_matches.append(
            ", ".join(
                sorted(
                    set(found)
                )
            )
        )

    df["health_keywords"] = (
        keyword_matches
    )

    candidates = df[
        df["health_keywords"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Total ChartQA pilot records:",
        len(df)
    )

    print(
        "Potential health-related records:",
        len(candidates)
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Keyword matches are only candidates."
    )

    print(
        "They are NOT ground-truth health labels."
    )

    print()
    print("=" * 70)
    print("PHASE 4D COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()