from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import requests
import xml.etree.ElementTree as ET


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"D:\downloads\misleading-health-infographic-detection"
)

PHASE11_DIR = PROJECT_ROOT / "outputs" / "phase11"
PHASE17_DIR = PROJECT_ROOT / "outputs" / "phase17"
PHASE19_DIR = PROJECT_ROOT / "outputs" / "phase19"


# ============================================================
# MODEL PATH
# ============================================================

MODEL_WEIGHTS = (
    PHASE11_DIR
    / "checkpoints"
    / "best_multimodal_model.weights.h5"
)

MODEL_COMPARISON = (
    PHASE17_DIR
    / "metrics"
    / "model_comparison.csv"
)


# ============================================================
# TEXT CONFIGURATION
# ============================================================

MAX_TOKENS = 20000
SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128


# ============================================================
# HEALTH VOCABULARY
# ============================================================

HEALTH_TERMS = {
    "health",
    "disease",
    "diseases",
    "diabetes",
    "cancer",
    "heart",
    "blood",
    "blood pressure",
    "cholesterol",
    "obesity",
    "weight",
    "vitamin",
    "mineral",
    "protein",
    "immune",
    "immunity",
    "infection",
    "inflammation",
    "medicine",
    "medication",
    "drug",
    "treatment",
    "therapy",
    "cure",
    "prevent",
    "prevention",
    "symptom",
    "symptoms",
    "doctor",
    "medical",
    "healthcare",
    "nutrition",
    "diet",
    "food",
    "supplement",
    "exercise",
    "sleep",
    "stress",
    "brain",
    "kidney",
    "liver",
    "lung",
    "pregnancy",
    "pregnant",
    "vaccine",
    "vaccination",
}


CLAIM_CUES = [
    "causes",
    "cause",
    "prevents",
    "prevent",
    "cures",
    "cure",
    "reduces",
    "increase",
    "increases",
    "decreases",
    "decrease",
    "lowers",
    "raises",
    "improves",
    "boosts",
    "protects",
    "leads to",
    "linked to",
    "associated with",
    "recommended",
    "should",
    "must",
    "avoid",
    "risk",
    "danger",
]


# ============================================================
# LOAD MODEL
# ============================================================

def build_multimodal_model():

    image_input = tf.keras.Input(
        shape=(224, 224, 3),
        name="image"
    )

    text_input = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="text"
    )

    # ---------------- IMAGE BRANCH ----------------

    image_encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    image_encoder.trainable = False

    image_features = image_encoder(image_input)

    image_features = tf.keras.layers.GlobalAveragePooling2D()(
        image_features
    )

    image_features = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(image_features)

    # ---------------- TEXT BRANCH ----------------

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )

    # The model architecture needs the vectorizer adapted to
    # training vocabulary. We retrieve it from the Phase 11
    # training split.

    train_csv = (
        PHASE11_DIR
        / "datasets"
        / "final_train.csv"
    )

    if train_csv.exists():

        train_df = pd.read_csv(train_csv)

        text_column = None

        for candidate in [
            "claim_text",
            "text",
            "ocr_text",
            "text_content"
        ]:
            if candidate in train_df.columns:
                text_column = candidate
                break

        if text_column is not None:
            vectorizer.adapt(
                train_df[text_column]
                .fillna("")
                .astype(str)
                .values
            )

    text_tokens = vectorizer(text_input)

    text_embedding = tf.keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )(text_tokens)

    text_features = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    )(text_embedding)

    text_features = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(text_features)

    # ---------------- FUSION ----------------

    combined = tf.keras.layers.Concatenate()(
        [image_features, text_features]
    )

    combined = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(combined)

    combined = tf.keras.layers.Dropout(0.4)(
        combined
    )

    combined = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(combined)

    combined = tf.keras.layers.Dropout(0.3)(
        combined
    )

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(combined)

    model = tf.keras.Model(
        inputs={
            "image": image_input,
            "text": text_input
        },
        outputs=output
    )

    return model


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    threshold = 0.5

    if MODEL_COMPARISON.exists():

        df = pd.read_csv(MODEL_COMPARISON)

        # Look for multimodal row
        possible_names = [
            "multimodal",
            "Multimodal",
            "multimodal_model",
        ]

        model_column = None

        for col in df.columns:
            if col.lower() in [
                "model",
                "model_name",
                "model_type"
            ]:
                model_column = col
                break

        if model_column is not None:

            mask = (
                df[model_column]
                .astype(str)
                .str.lower()
                .str.contains("multimodal")
            )

            matching = df[mask]

            if not matching.empty:

                for col in [
                    "threshold",
                    "best_threshold",
                    "validation_threshold"
                ]:

                    if col in matching.columns:

                        threshold = float(
                            matching.iloc[0][col]
                        )

                        break

    return threshold


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    # EfficientNet preprocessing is built into the
    # TensorFlow EfficientNet implementation.
    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# OCR
# ============================================================

def extract_ocr(image_path):

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False
    )

    result = ocr.predict(
        str(image_path)
    )

    texts = []

    def recursive_extract(obj):

        if isinstance(obj, dict):

            for key, value in obj.items():

                if key in [
                    "rec_texts",
                    "text",
                    "texts"
                ]:

                    if isinstance(value, list):

                        for item in value:

                            if isinstance(item, str):
                                texts.append(item)

                recursive_extract(value)

        elif isinstance(obj, list):

            for item in obj:
                recursive_extract(item)

    recursive_extract(result)

    cleaned = []

    seen = set()

    for text in texts:

        text = str(text).strip()

        if text and text not in seen:

            cleaned.append(text)
            seen.add(text)

    return " ".join(cleaned)


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 15:
            continue

        lower = sentence.lower()

        health_score = sum(
            term in lower
            for term in HEALTH_TERMS
        )

        cue_score = sum(
            cue in lower
            for cue in CLAIM_CUES
        )

        quantitative = re.findall(
            r"\b\d+(?:\.\d+)?\s*%?\b",
            sentence
        )

        if (
            health_score > 0
            and (
                cue_score > 0
                or quantitative
            )
        ):

            claims.append({
                "claim_text": sentence,
                "health_term_count": health_score,
                "claim_cue_count": cue_score,
                "quantitative_values": quantitative,
            })

    # If no sentence passes the rules,
    # use the OCR text itself as a fallback.

    if not claims and len(text) >= 15:

        claims.append({
            "claim_text": text[:1000],
            "health_term_count": 0,
            "claim_cue_count": 0,
            "quantitative_values": [],
        })

    return claims

# ============================================================
# PUBMED EVIDENCE RETRIEVAL
# ============================================================

def retrieve_pubmed_evidence(
    claim_text,
    max_results=5
):
    """
    Retrieve PubMed articles related to a detected
    health claim.

    Note:
    Retrieval relevance does not establish that
    the claim is true or false.
    """

    try:

        # ----------------------------------------------------
        # STEP 1: Search PubMed
        # ----------------------------------------------------

        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/"
            "entrez/eutils/esearch.fcgi"
        )

        search_params = {
            "db": "pubmed",
            "term": claim_text,
            "retmode": "json",
            "retmax": max_results,
        }

        search_response = requests.get(
            search_url,
            params=search_params,
            timeout=30
        )

        search_response.raise_for_status()

        search_data = search_response.json()

        pmids = (
            search_data
            .get("esearchresult", {})
            .get("idlist", [])
        )

        # No PubMed results
        if not pmids:
            return []

        # ----------------------------------------------------
        # STEP 2: Fetch article details
        # ----------------------------------------------------

        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/"
            "entrez/eutils/efetch.fcgi"
        )

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        fetch_response = requests.get(
            fetch_url,
            params=fetch_params,
            timeout=30
        )

        fetch_response.raise_for_status()

        root = ET.fromstring(
            fetch_response.text
        )

        # ----------------------------------------------------
        # STEP 3: Extract article information
        # ----------------------------------------------------

        evidence = []

        for article in root.findall(
            ".//PubmedArticle"
        ):

            # ---------------- PMID ----------------

            pmid_element = article.find(
                ".//PMID"
            )

            pmid = (
                pmid_element.text
                if pmid_element is not None
                else ""
            )

            # ---------------- TITLE ----------------

            title_element = article.find(
                ".//ArticleTitle"
            )

            if title_element is not None:

                title = "".join(
                    title_element.itertext()
                )

            else:

                title = ""

            # ---------------- ABSTRACT ----------------

            abstract_elements = article.findall(
                ".//AbstractText"
            )

            abstract_parts = []

            for element in abstract_elements:

                abstract_parts.append(
                    "".join(
                        element.itertext()
                    )
                )

            abstract = " ".join(
                abstract_parts
            )

            # ---------------- SAVE ARTICLE ----------------

            evidence.append({

                "pmid": pmid,

                "title": title,

                "abstract": abstract,

                "source": "PubMed",

                "url":
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        return evidence

    except Exception as e:

        return [
            {
                "error": str(e),
                "source": "PubMed"
            }
        ]


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_infographic(
    model,
    image_path,
    claim_text,
    threshold
):

    image = preprocess_image(
        image_path
    )

    text = np.array(
        [claim_text],
        dtype=object
    )

    probability = float(
        model.predict(
            {
                "image": image,
                "text": text
            },
            verbose=0
        )[0][0]
    )

    if probability >= threshold:

        classification = "Misleading"

    else:

        classification = "Non-misleading"

    return probability, classification


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_inference(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ---------------- OCR ----------------

    ocr_text = extract_ocr(
        image_path
    )

    # ---------------- CLAIMS ----------------

    claims = extract_claims(
        ocr_text
    )

    # ---------------- MODEL ----------------

    model = build_multimodal_model()

    model.load_weights(
        str(MODEL_WEIGHTS)
    )

    threshold = load_threshold()

    results = []

    # ---------------- PREDICTION ----------------

    for index, claim in enumerate(
        claims,
        start=1
    ):

        probability, classification = (
            predict_infographic(
                model,
                image_path,
                claim["claim_text"],
                threshold
            )
        )
        
        evidence = retrieve_pubmed_evidence(
            claim["claim_text"],
            max_results=5
        )

        results.append({

            "claim_id": f"claim_{index}",

            "claim_text":
                claim["claim_text"],
        
            "misleading_probability":
                probability,
        
            "classification":
                classification,
        
            "threshold":
                threshold,
        
            "quantitative_values":
                claim["quantitative_values"],
        
            "evidence":
                evidence,
        })

    # ---------------- SAVE ----------------

    PHASE19_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "image":
            str(image_path),

        "ocr_text":
            ocr_text,

        "threshold":
            threshold,

        "claims":
            results,
    }

    output_path = (
        PHASE19_DIR
        / "inference"
        / "result.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output