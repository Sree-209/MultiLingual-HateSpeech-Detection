import streamlit as st

st.set_page_config(page_title="Hate Speech Detector", layout="centered")

import subprocess
import time
import requests

# Automatically start FastAPI backend
@st.cache_resource
def start_backend():
    return subprocess.Popen(
        ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

start_backend()

# Streamlit UI
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
                st.error("Backend is not responding. Please check if FastAPI started properly.")
