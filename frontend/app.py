import streamlit as st
import requests

# ✅ This must always be first
st.set_page_config(page_title="Hate Speech Detector", layout="centered")

# 🔠 UI layout
st.title("Multilingual Hate Speech Detector")
st.markdown("Enter a code-mixed (Hinglish) comment to check if it's **Hate** or **Non-Hate**.")

user_input = st.text_area("Enter your comment below", height=120)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            try:
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
                    st.error("Failed to get a valid response from backend.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend is not running. Please make sure to start it before launching this app.")
