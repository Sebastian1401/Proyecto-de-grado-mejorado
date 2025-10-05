import os, io, json
import cv2
import numpy as np
from typing import List, Dict, Optional
from app.config import settings

class StorageFS:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = base_dir or settings.PATIENTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def patient_dir(self, cedula: str) -> str:
        p = os.path.join(self.base_dir, cedula)
        os.makedirs(p, exist_ok=True)
        return p

    def save_text(self, cedula: str, filename: str, content: str) -> None:
        with open(os.path.join(self.patient_dir(cedula), filename), "w", encoding="utf-8") as f:
            f.write(content)

    def read_text(self, cedula: str, filename: str) -> Optional[str]:
        path = os.path.join(self.patient_dir(cedula), filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write_image_from_np(self, cedula: str, filename: str, img: np.ndarray) -> str:
        full = os.path.join(self.patient_dir(cedula), filename)
        cv2.imwrite(full, img)
        return full

    def list_images(self, cedula: str, exts=(".jpg", ".jpeg", ".png")) -> List[str]:
        pdir = self.patient_dir(cedula)
        return sorted([f for f in os.listdir(pdir) if f.lower().endswith(exts)])

    def delete_file(self, cedula: str, filename: str) -> bool:
        if ".." in filename or filename.startswith("/"):
            return False
        full = os.path.join(self.patient_dir(cedula), filename)
        if os.path.exists(full):
            os.remove(full)
            return True
        return False

    def file_path(self, cedula: str, filename: str) -> str:
        return os.path.join(self.patient_dir(cedula), filename)

    def list_patients(self) -> List[str]:
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]