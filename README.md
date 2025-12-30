Markdown

# Rebar Counting – Setup Guide

This project is a Streamlit app for counting rebars using an ONNX model.  
Below are only the steps to **clone**, **set up**, and **run** the app.

---

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPO_URL> rebar-counting
cd rebar-counting
Replace <YOUR_GITHUB_REPO_URL> with your actual repository URL.

2. Create and Activate a Virtual Environment
On Windows (PowerShell)
Bash

python -m venv .venv
.venv\Scripts\Activate.ps1
On Linux / macOS
Bash

python -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Make sure you are in the rebar-counting folder and the virtual environment is active.

Bash

pip install -r requirements.txt
4. Add Your ONNX Model
Place your model file (e.g. Yolo11m.onnx) in the models/ folder:

text

rebar-counting/
├─ models/
│   └─ Yolo11m.onnx   ← put your ONNX model here
If the filename is different, open config.py and update:

Python

MODEL_PATH = os.path.join(BASE_DIR, "models", "YOUR_MODEL_NAME.onnx")
5. Run the App
From the project root (rebar-counting/), with the virtual environment active:

Bash

streamlit run app.py
Streamlit will print a local URL (usually http://localhost:8501).
Open that URL in your browser to use the app.

