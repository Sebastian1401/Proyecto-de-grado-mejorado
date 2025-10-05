# app/config/settings.py
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PATIENTS_DIR = os.environ.get("PATIENTS_DIR", "var/patients")

MODELS_DIR = os.environ.get("MODELS_DIR", "app/models")
DATA_DIR   = os.environ.get("DATA_DIR", "app/config")

RKNN_MODEL_PATH = os.path.join(MODELS_DIR, "model1.rknn")
CLASSES_YAML    = os.path.join(DATA_DIR, "data.yaml")

RKNN_IMG_SIZE = int(os.environ.get("RKNN_IMG_SIZE", 640))