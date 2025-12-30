# Rebar Counting – Streamlit + ONNX + DepthAI

A modern, full-stack computer vision app for **automatic rebar counting**:

- Web UI built with **Streamlit**
- Object detection via **ONNX Runtime** (YOLO-style model)
- Live input from:
  - **IP webcam** (MJPEG + snapshot URL)
  - **OAK‑D Pro** camera via **DepthAI**
- User accounts with **login / signup**
- Persistent **history log** with thumbnails and pagination

---

## Features

### Rebar Detection

- Uses a YOLO-style ONNX model (e.g. `Yolo11m.onnx`)
- Flexible input size (reads from the ONNX model; supports 640 or 1024 or dynamic)
- Bounding boxes drawn on detected rebars
- Output composited into a **1080p frame with a header** showing total count

### Live Sources

- **IP Webcam**
  - Configurable MJPEG video URL
  - Configurable JPEG snapshot URL (used for Capture & Count)
- **OAK‑D Pro (DepthAI)**
  - Live RGB frames from OAK‑D Pro
  - Capture & Count from the depth camera stream

### User Accounts

- Signup / Login with:
  - Username
  - Email
  - Password (PBKDF2-HMAC-SHA256)
- Credentials stored in **SQLite**
- “Remember me” via a secure token stored in `data/session.json`

### Detection History

- All detections logged in `detections` table:
  - Timestamp (UTC, rendered as local time)
  - Stream source (IP webcam URL / OAK‑D / Upload)
  - Paths to full image + thumbnail
  - Rebar count, width, height
- History page:
  - Paginated table
  - Inline thumbnail preview
  - View full image
  - Delete detection

### UI/UX

- Single-page style dashboard in Streamlit:
  - Beautiful gradient theme
  - Navigation bar (Dashboard / History / Logout)
  - Live view panel
  - Capture & Count controls
  - Upload image tab
  - Recent capture gallery (cards with thumbnails)

---

## Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Model Inference**: [ONNX Runtime](https://onnxruntime.ai/)
- **Computer Vision**: [OpenCV](https://opencv.org/), `numpy`
- **Camera**: [DepthAI](https://docs.luxonis.com/) for OAK‑D Pro
- **Backend / DB**: SQLite (via `sqlite3`)
- **Auth & Security**:
  - Password hashing (PBKDF2-HMAC-SHA256)
  - HMAC-based “remember me” tokens

---

## Project Structure

```text
rebar-counting/              # ← Open this folder in VS Code
├─ app.py                    # Entry point: streamlit run app.py
├─ auth.py                   # Authentication & session token handling
├─ config.py                 # Paths, constants, pagination config
├─ db.py                     # SQLite connection and schema creation
├─ detector.py               # ONNX model loading + detection + detection DB helpers
├─ helpers.py                # Generic Streamlit helpers (rerun, flash, etc.)
├─ oak_utils.py              # OAK‑D Pro / DepthAI pipeline controls
├─ pages.py                  # Streamlit page functions (login, signup, dashboard, history)
├─ style.py                  # Global CSS theme for Streamlit
├─ utils.py                  # Time formatting and small utility functions
│
├─ models/
│   └─ Yolo11m.onnx          # Your ONNX model (change name/path in config if needed)
│
├─ data/                     # Created at runtime
│   ├─ app.db                # SQLite DB
│   ├─ detections/           # Stored full-size detection images
│   ├─ thumbs/               # Thumbnails for history page
│   └─ session.json          # “Remember me” token
│
├─ .vscode/
│   ├─ launch.json           # VS Code Run/Debug configuration
│   ├─ settings.json         # Python / formatter / exclude settings
│   └─ extensions.json       # Recommended VS Code extensions (optional)
│
├─ requirements.txt          # Python dependencies
├─ .gitignore                # Git ignore rules
└─ README.md                 # This file
