"""Service: pacientes (datos y capturas) sobre filesystem adapter."""
import os
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import cv2
from app.adapters.storage_fs import StorageFS

class PatientService:
    def __init__(self, storage: StorageFS | None = None) -> None:
        self.storage = storage or StorageFS()
        self.info_file = "datos_paciente.txt"

    def save_patient_info(self, datos: Dict[str, str]) -> None:
        cedula = datos.get("Cédula") or datos.get("cedula")
        assert cedula, "Cédula requerida"
        body = "\n".join([f"{k}: {v}" for k, v in datos.items()]) + "\n"
        self.storage.save_text(cedula, self.info_file, body)

    def get_patient_info(self, cedula: str) -> Dict[str, str]:
        txt = self.storage.read_text(cedula, self.info_file)
        info: Dict[str, str] = {}
        if txt:
            for line in txt.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
        if "Cédula" not in info:
            info["Cédula"] = cedula
        return info

    def list_captures(self, cedula: str) -> List[str]:
        return self.storage.list_images(cedula)

    def delete_capture(self, cedula: str, filename: str) -> bool:
        return self.storage.delete_file(cedula, filename)

    def save_capture_blob(self, cedula: str, np_image: np.ndarray) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captura_{ts}.jpg"
        self.storage.write_image_from_np(cedula, filename, np_image)
        return filename

    def list_patients_summary(self) -> List[Dict[str, str]]:
        res = []
        for ced in self.storage.list_patients():
            info = self.get_patient_info(ced)
            res.append({
                "nombre": info.get("Nombre", "N/A"),
                "cedula": ced,
                "edad": info.get("Edad", "N/A"),
                "genero": info.get("Género", "N/A"),
                "antecedentes": info.get("Antecedentes", "N/A"),
            })
        return res