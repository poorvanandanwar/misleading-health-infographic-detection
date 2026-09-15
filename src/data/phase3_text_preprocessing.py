from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PUBHEALTH_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase2"
    / "pubhealth"
    / "pubhealth_clean.csv"
)

HEALTHFC_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase2"
    / "healthfc"
    / "healthfc_clean.csv"
)

PUBHEALTH_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase3"
    / "pubhealth"
)

HEALTHFC_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase3"
    / "healthfc"
)

INTERIM_DIR = (
    ROOT
    / "data"
    / "interim"
    / "phase3"
)


PUBHEALTH_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HEALTHFC_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize line breaks
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text):

    if not text:
        return 0

    return len(text.split())


# ============================================================
# PUBHEALTH
# ============================================================

def process_pubhealth():

    print("\n" + "=" * 70)
    print("PUBHEALTH TEXT PREPROCESSING")
    print("=" * 70)

    print("\nInput:")
    print(PUBHEALTH_INPUT)

    df = pd.read_csv(PUBHEALTH_INPUT)

    print("\nOriginal shape:", df.shape)

    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    text_columns = [
        "claim",
        "explanation",
        "main_text",
        "sources",
        "subjects",
        "fact_checkers"
    ]

    for column in text_columns:

        if column in df.columns:

            clean_column = f"{column}_clean"

            df[clean_column] = (
                df[column]
                .apply(clean_text)
            )

    # --------------------------------------------------------
    # Add text statistics
    # --------------------------------------------------------

    df["claim_char_count"] = (
        df["claim_clean"]
        .str.len()
    )

    df["claim_word_count"] = (
        df["claim_clean"]
        .apply(word_count)
    )

    df["explanation_word_count"] = (
        df["explanation_clean"]
        .apply(word_count)
    )

    df["main_text_word_count"] = (
        df["main_text_clean"]
        .apply(word_count)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PUBHEALTH_OUTPUT_DIR
        / "pubhealth_text_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(output_path)

    print("\nFinal shape:", df.shape)

    print("\nLabel distribution:")

    if "label" in df.columns:
        print(
            df["label"]
            .value_counts(dropna=False)
        )

    print("\nMissing cleaned claims:")

    print(
        (df["claim_clean"] == "").sum()
    )

    return df


# ============================================================
# HEALTHFC
# ============================================================

def process_healthfc():

    print("\n" + "=" * 70)
    print("HEALTHFC TEXT PREPROCESSING")
    print("=" * 70)

    print("\nInput:")
    print(HEALTHFC_INPUT)

    df = pd.read_csv(HEALTHFC_INPUT)

    print("\nOriginal shape:", df.shape)

    # --------------------------------------------------------
    # Clean English columns
    # --------------------------------------------------------

    english_columns = [
        "en_claim",
        "en_explanation",
        "en_top_sentences"
    ]

    for column in english_columns:

        if column in df.columns:

            clean_column = f"{column}_clean"

            df[clean_column] = (
                df[column]
                .apply(clean_text)
            )

    # --------------------------------------------------------
    # Clean German / metadata columns
    # --------------------------------------------------------

    other_columns = [
        "de_claim",
        "de_explanation",
        "de_top_sentences",
        "de_verdict",
        "de_title",
        "authors"
    ]

    for column in other_columns:

        if column in df.columns:

            clean_column = f"{column}_clean"

            df[clean_column] = (
                df[column]
                .apply(clean_text)
            )

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    df["claim_char_count"] = (
        df["en_claim_clean"]
        .str.len()
    )

    df["claim_word_count"] = (
        df["en_claim_clean"]
        .apply(word_count)
    )

    df["explanation_word_count"] = (
        df["en_explanation_clean"]
        .apply(word_count)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        HEALTHFC_OUTPUT_DIR
        / "healthfc_text_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(output_path)

    print("\nFinal shape:", df.shape)

    print("\nLabel distribution:")

    if "label" in df.columns:
        print(
            df["label"]
            .value_counts(dropna=False)
        )

    print("\nEmpty English claims:")

    print(
        (df["en_claim_clean"] == "").sum()
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3B - TEXT PREPROCESSING")
    print("=" * 70)

    pubhealth_df = process_pubhealth()

    healthfc_df = process_healthfc()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "dataset": [
            "PubHealth",
            "HealthFC"
        ],
        "rows": [
            len(pubhealth_df),
            len(healthfc_df)
        ],
        "empty_claims": [
            (pubhealth_df["claim_clean"] == "").sum(),
            (healthfc_df["en_claim_clean"] == "").sum()
        ]
    })

    summary_path = (
        INTERIM_DIR
        / "text_preprocessing_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("PHASE 3B COMPLETE")
    print("=" * 70)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()