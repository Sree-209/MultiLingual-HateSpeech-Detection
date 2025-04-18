import subprocess
import time
import webbrowser

# ✅ Start FastAPI backend
backend = subprocess.Popen(
    ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Wait for backend to spin up
time.sleep(5)

# Open browser to Streamlit manually
webbrowser.open("http://localhost:8501")

# Start Streamlit frontend
subprocess.run(["streamlit", "run", "frontend/app.py"])
# NOTE: this blocks until Streamlit exits

# Kill backend when Streamlit closes
backend.terminate()
