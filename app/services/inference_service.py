# app/services/inference_service.py
"""Service: inference (coordina el adapter RKNN y estandariza salida)."""

import cv2
import numpy as np
from typing import List, Dict
from app.adapters.rknn_adapter import RknnModel
from app.config import settings


class InferenceService:
    def __init__(
        self,
        model_path: str | None = None,
        yaml_path: str | None = None,
        img_size: int | None = None,
    ) -> None:
        model_path = model_path or settings.RKNN_MODEL_PATH
        yaml_path  = yaml_path  or settings.CLASSES_YAML
        img_size   = int(img_size or settings.RKNN_IMG_SIZE)

        # Adapter RKNN existente (tu wrapper actual)
        self.model = RknnModel(model_path=model_path, yaml_path=yaml_path, img_size=img_size)
        self.img_size = img_size

        # Grupos de clases (misma lógica que tenías)
        self.grupos = {
            "MALIGNO/PREMALIGNO": ["AKIEC", "BCC", "SCC", "MEL"],
            "BENIGNO": ["BKL", "DF", "NV", "VASC"],
        }

    def predict(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Devuelve detecciones normalizadas (en coords 640x640) usando el adapter RKNN."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = self.model.predict(img_rgb)  # [{class_id,name,confidence,bbox_xyxy}]
        return dets

    def label_for_class(self, class_name: str) -> str:
        if class_name in self.grupos["MALIGNO/PREMALIGNO"]:
            return "MALIGNO"
        if class_name in self.grupos["BENIGNO"]:
            return "BENIGNO"
        return class_name

    def adjust_conf(self, conf: float) -> float:
        if conf < 0.3:
            return conf + 0.3
        if conf < 0.4:
            return conf + 0.2
        if conf < 0.5:
            return conf + 0.1
        return conf