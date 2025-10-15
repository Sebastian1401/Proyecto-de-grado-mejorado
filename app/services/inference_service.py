# app/services/inference_service.py
import cv2
import numpy as np
from typing import List, Dict
from app.adapters.rknn_adapter import RknnModel
from app.config import settings
from app.services.settings_service import Thresholds  # <- nuevo import

class InferenceService:
    def __init__(self, model_path: str | None = None, yaml_path: str | None = None, img_size: int | None = None) -> None:
        model_path = model_path or settings.RKNN_MODEL_PATH
        yaml_path  = yaml_path  or settings.CLASSES_YAML
        img_size   = int(img_size or settings.RKNN_IMG_SIZE)
        self.model = RknnModel(model_path=model_path, yaml_path=yaml_path, img_size=img_size)
        self.img_size = img_size
        self.grupos = {
            "MALIGNO/PREMALIGNO": ["AKIEC", "BCC", "SCC", "MEL"],
            "BENIGNO": ["BKL", "DF", "NV", "VASC"],
        }

    def predict(self, frame_bgr: np.ndarray, thr: Thresholds | None = None) -> list[dict]:
        """Inferencia; si 'thr' es None, el adapter usará sus defaults."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if hasattr(self.model, "preprocess") and hasattr(self.model, "rknn"):
            img_input = self.model.preprocess(img_rgb)
            outputs = self.model.rknn.inference(inputs=[img_input])
            try:
                if thr is not None:
                    return self.model.postprocess(outputs, float(thr.conf_th), float(thr.iou_th), float(thr.min_box_frac))
                else:
                    return self.model.postprocess(outputs)
            except TypeError:
                try:
                    return self.model.postprocess(outputs)
                except Exception:
                    return self.model.predict(img_rgb)
        try:
            return self.model.predict(img_rgb)
        except Exception:
            return []

    def label_for_class(self, class_name: str) -> str:
        if class_name in self.grupos["MALIGNO/PREMALIGNO"]:
            return "MALIGNO"
        if class_name in self.grupos["BENIGNO"]:
            return "BENIGNO"
        return class_name

    def adjust_conf(self, conf: float) -> float:
        if conf < 0.3: return conf + 0.5
        if conf < 0.4: return conf + 0.4
        if conf < 0.5: return conf + 0.3
        if conf < 0.6: return conf + 0.2
        if conf < 0.7: return conf + 0.1
        return conf
