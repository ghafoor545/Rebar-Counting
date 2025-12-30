Markdown

# Rebar Counting App

Streamlit application for counting rebars using an ONNX object detection model,
with support for IP camera and OAK-D Pro (DepthAI).

## Features

- User signup / login with SQLite backend.
- ONNXRuntime-based detection (YOLO-style).
- Live view:
  - IP camera (MJPEG + snapshot URL).
  - OAK-D Pro via DepthAI.
- Capture & Count from live feed or uploaded images.
- Detection history with thumbnails and pagination.

## Project Structure

```text
rebar-counting/
├─ app.py
├─ auth.py
├─ config.py
├─ db.py
├─ detector.py
├─ helpers.py
├─ oak_utils.py
├─ pages.py
├─ style.py
├─ utils.py
├─ models/
│   └─ Yolo11m.onnx
├─ data/
├─ .vscode/
├─ requirements.txt
├─ .gitignore
└─ README.md
Setup
Bash

git clone <this-repo>
cd rebar-counting

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
Place your ONNX model at:

text

models/Yolo11m.onnx
(or update MODEL_PATH in config.py).

Run
Bash

streamlit run app.py
Open the URL shown in the terminal (usually http://localhost:8501).

text


---

After creating all these files and placing your model at `models/Yolo11m.onnx`, you can:

```bash
cd rebar-counting
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py