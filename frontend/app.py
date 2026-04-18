import streamlit as st
import requests

st.set_page_config(page_title="Hallucination Detector")

st.title("🧠 Hallucination Detector")

text = st.text_area("Enter your text here:")

if st.button("Check"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Result:")
                st.write("**Prediction:**", result["result"])
                st.write("**Confidence:**", result["confidence"])
                st.write("**Evidence:**", result["evidence"])

            else:
                st.error("API error")

        except:
            st.error("Cannot connect to API. Is backend running?")