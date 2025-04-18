import streamlit as st
import requests

st.set_page_config(page_title="Hate Speech Detector", layout="centered")

st.title("Multilingual Hate Speech Detector")
st.markdown("Enter a code-mixed (Hinglish) comment to check if it's **Hate** or **Non-Hate**.")

user_input = st.text_area("Enter your comment below", height=120)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            response = requests.post(
                "http://localhost:8000/predict/",
                json={"text": user_input}
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction: **{data['prediction']}**")
                st.progress(data["confidence"])
                st.caption(f"Confidence: {round(data['confidence'] * 100, 2)}%")
            else:
                st.error("Failed to get response from backend.")
