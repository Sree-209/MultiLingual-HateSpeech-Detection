import subprocess
import time
import os

# Start the backend server
backend = subprocess.Popen(
    ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
)

# Give it some time to start
time.sleep(5)

# Start the Streamlit app
subprocess.run(["streamlit", "run", "frontend/app.py"])

# When Streamlit exits, kill the backend
backend.terminate()
