import streamlit as st
import subprocess

st.set_page_config(page_title="Launcher", layout="centered")
st.title("Launch Multilingual Hate Speech App")

st.markdown("Click the button below to start both the backend (FastAPI) and the frontend (Streamlit app).")

if st.button("Launch App"):
    subprocess.Popen(["python3", "launch.py"])
    st.success("Launched successfully! Check your browser.")
