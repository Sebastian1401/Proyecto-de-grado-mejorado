import cv2
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite
import yaml


class RknnModel:
    def __init__(self, model_path="weights/model1.rknn", yaml_path="data/data.yaml", img_size=640):
        self.model_path = Path(model_path)
        self.yaml_path = Path(yaml_path)
        self.img_size = img_size

        # Cargar nombres de clases
        with open(self.yaml_path, "r") as f:
            self.class_names = yaml.safe_load(f)["names"]

        # Iniciar RKNN
        self.rknn = RKNNLite(verbose=False)
        print(f"[RKNN] Cargando modelo: {self.model_path}")
        if self.rknn.load_rknn(str(self.model_path)) != 0:
            raise RuntimeError("Error cargando modelo RKNN")

        print("[RKNN] Inicializando runtime...")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("Error inicializando runtime RKNN")

        print("[RKNN] NPU listo.")

    def preprocess(self, img):
        """Redimensiona y prepara la imagen"""
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0)  # [1,H,W,3]
        return img_input

    def postprocess(self, outputs, conf_th=0.6, iou_th=0.3, min_box_frac=0.003):
        """Convierte la salida de la NPU en cajas + clases"""
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]
        pred = pred.astype(np.float32)

        xywh = pred[:, :4]
        obj = pred[:, 4]
        cls = pred[:, 5:]

        # Sigmoid si hace falta
        if np.nanmax(obj) > 1 or np.nanmin(obj) < 0:
            obj = 1 / (1 + np.exp(-obj))
        if np.nanmax(cls) > 1 or np.nanmin(cls) < 0:
            cls = 1 / (1 + np.exp(-cls))

        cls_ids = np.argmax(cls, axis=1)
        cls_conf = cls[np.arange(cls.shape[0]), cls_ids]
        scores = obj * cls_conf

        # Filtrado por confianza
        keep = scores >= conf_th
        if not np.any(keep):
            return []

        xywh, scores, cls_ids = xywh[keep], scores[keep], cls_ids[keep]

        # Convertir a XYXY
        x_c, y_c, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Filtrar cajitas muy pequeñas
        min_area = (self.img_size * self.img_size) * min_box_frac
        areas = w * h
        big = areas >= min_area
        if not np.any(big):
            return []

        boxes, scores, cls_ids = boxes[big], scores[big], cls_ids[big]

        # Clip límites
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.img_size - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.img_size - 1)

        # Empaquetar resultados
        detections = []
        for b, s, c in zip(boxes, scores, cls_ids):
            detections.append({
                "class_id": int(c),
                "class_name": self.class_names[int(c)],
                "confidence": float(s),
                "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            })

        return detections

    def predict(self, img):
        """Pipeline completo: pre → inferencia → post"""
        img_input = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self.postprocess(outputs)


# Prueba rápida (solo cuando se ejecute directamente este archivo)
if __name__ == "__main__":
    model = RknnModel()
    img = cv2.imread("imagen_de_prueba.jpg")
    detections = model.predict(img)
    print("Detecciones:", detections)