import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ONNX model path (models/Yolo11m.onnx)
MODEL_PATH = os.path.join(BASE_DIR, "models", "Yolo11m.onnx")

# App secret for "remember me" token
APP_SECRET = "change-this-secret"

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
DET_DIR = os.path.join(DATA_DIR, "detections")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")

# History pagination
PER_PAGE = 8

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DET_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
