def apply_styles(st):

    st.markdown(
        """
        <style>

        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-top: 20px;
        }

        .claim-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )