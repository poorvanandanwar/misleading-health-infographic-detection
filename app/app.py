import sys
from pathlib import Path
import json

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"D:\downloads\misleading-health-infographic-detection"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from app.inference import run_inference
from app.styles import apply_styles


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Infographic Analyzer",
    page_icon="🔍",
    layout="wide"
)

apply_styles(st)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI Health Infographic Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect potentially misleading health claims '
    'using OCR, multimodal classification, '
    'evidence retrieval and explainability.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application is the Phase 19 deployment
        of the misleading health infographic detection
        pipeline.
        """
    )

    st.markdown(
        """
        **Pipeline**

        1. Image upload
        2. OCR
        3. Claim extraction
        4. Multimodal classification
        5. Evidence retrieval
        6. Explanation
        """
    )

    st.divider()

    st.caption(
        "Phase 19 — Streamlit Application"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "Upload Infographic"
)

uploaded_file = st.file_uploader(
    "Choose a health infographic",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------------

    st.image(
        uploaded_file,
        caption="Uploaded infographic",
        use_container_width=True
    )

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    phase19_input_dir = (
        PROJECT_ROOT
        / "outputs"
        / "phase19"
        / "inference"
    )

    phase19_input_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        phase19_input_dir
        / uploaded_file.name
    )

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.divider()

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Infographic",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.status(
                "Running analysis...",
                expanded=True
            ) as status:

                st.write(
                    "Running OCR..."
                )

                result = run_inference(
                    image_path
                )

                st.write(
                    "Extracting health claims..."
                )

                st.write(
                    "Running multimodal classification..."
                )

                st.write(
                    "Retrieving PubMed evidence..."
                )

                st.write(
                    "Preparing explanation..."
                )

                status.update(
                    label="Analysis completed!",
                    state="complete"
                )

            st.session_state[
                "result"
            ] = result

        except Exception as e:

            st.error(
                "An error occurred during inference."
            )

            st.exception(e)


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state[
        "result"
    ]

    st.divider()

    st.header(
        "Analysis Results"
    )


    # ========================================================
    # OCR
    # ========================================================

    st.subheader(
        "1. Extracted Text"
    )

    ocr_text = result.get(
        "ocr_text",
        ""
    )

    if ocr_text:

        st.text_area(
            "OCR output",
            ocr_text,
            height=180
        )

    else:

        st.warning(
            "No readable text was detected."
        )


    # ========================================================
    # CLAIMS
    # ========================================================

    st.subheader(
        "2. Detected Health Claims"
    )

    claims = result.get(
        "claims",
        []
    )

    if not claims:

        st.info(
            "No health claims were detected."
        )

    else:

        for index, claim in enumerate(
            claims,
            start=1
        ):

            st.markdown(
                f"### Claim {index}"
            )

            st.markdown(
                '<div class="claim-box">',
                unsafe_allow_html=True
            )

            st.write(
                "**Claim:**"
            )

            st.write(
                claim.get(
                    "claim_text",
                    ""
                )
            )

            probability = float(
                claim.get(
                    "misleading_probability",
                    0
                )
            )

            classification = claim.get(
                "classification",
                "Unknown"
            )

            threshold = float(
                claim.get(
                    "threshold",
                    0.5
                )
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Misleading Probability",
                    f"{probability:.3f}"
                )

            with c2:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.3f}"
                )

            with c3:

                st.metric(
                    "Classification",
                    classification
                )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            if (
                classification.lower()
                == "misleading"
            ):

                st.warning(
                    "The multimodal model classified "
                    "this claim as potentially misleading "
                    "at the selected validation threshold."
                )

            else:

                st.success(
                    "The multimodal model classified "
                    "this claim as non-misleading "
                    "at the selected validation threshold."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # EVIDENCE
            # =================================================

            st.subheader(
                "3. Retrieved Evidence"
            )

            evidence = claim.get(
                "evidence",
                []
            )

            if not evidence:

                st.info(
                    "No PubMed evidence was retrieved "
                    "for this claim."
                )

            else:

                valid_evidence = [
                    item
                    for item in evidence
                    if "error" not in item
                ]

                if not valid_evidence:

                    st.warning(
                        "PubMed retrieval returned an error."
                    )

                for evidence_index, item in enumerate(
                    valid_evidence,
                    start=1
                ):

                    title = item.get(
                        "title",
                        "Untitled PubMed article"
                    )

                    pmid = item.get(
                        "pmid",
                        ""
                    )

                    abstract = item.get(
                        "abstract",
                        ""
                    )

                    with st.expander(
                        f"Evidence {evidence_index}: {title}"
                    ):

                        st.write(
                            "**Source:** PubMed"
                        )

                        if pmid:

                            st.write(
                                f"**PMID:** {pmid}"
                            )

                        if abstract:

                            st.write(
                                "**Abstract:**"
                            )

                            st.write(
                                abstract
                            )

                        else:

                            st.info(
                                "No abstract was available."
                            )

                        if pmid:

                            st.markdown(
                                f"[Open PubMed article]"
                                f"(https://pubmed.ncbi.nlm.nih.gov/"
                                f"{pmid}/)"
                            )

            st.caption(
                "Evidence retrieval identifies relevant "
                "literature for analysis. Retrieved evidence "
                "does not by itself establish that a health "
                "claim is true or false."
            )

            st.divider()


    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================

    st.subheader(
        "4. Download Analysis"
    )

    result_json = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Complete JSON",
        data=result_json,
        file_name="phase19_result.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Misleading Health Infographic Detection "
    "— Phase 19"
)