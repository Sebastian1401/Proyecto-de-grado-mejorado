import cv2
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite
import yaml


class RknnModel:
    """
    Wrapper ligero para inferencia RKNN + postproceso estilo YOLO.
    Incluye 'perillas' ajustables en runtime:
        - conf_th: umbral de confianza
        - iou_th:  umbral de NMS (class-agnostic)
        - min_box_frac: área mínima relativa a (img_size^2)
    """
    def __init__(
        self,
        model_path="weights/model1.rknn",
        yaml_path="data/data.yaml",
        img_size=640,
        conf_th=0.60,
        iou_th=0.30,
        min_box_frac=0.003,
        nms_topk=300
    ):
        self.model_path = Path(model_path)
        self.yaml_path = Path(yaml_path)
        self.img_size = int(img_size)

        # perillas (runtime-configurable)
        self.conf_th = float(conf_th)
        self.iou_th = float(iou_th)
        self.min_box_frac = float(min_box_frac)
        self.nms_topk = int(nms_topk)

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

    # -------- perillas (setters/getters) --------
    def set_thresholds(self, conf_th=None, iou_th=None, min_box_frac=None):
        if conf_th is not None: self.conf_th = float(conf_th)
        if iou_th is not None: self.iou_th = float(iou_th)
        if min_box_frac is not None: self.min_box_frac = float(min_box_frac)

    def get_thresholds(self):
        return {
            "conf_th": self.conf_th,
            "iou_th": self.iou_th,
            "min_box_frac": self.min_box_frac
        }

    # -------- pre/post/inferencia --------
    def preprocess(self, img):
        """
        Espera img en RGB (app.py ya convierte).
        Redimensiona a (img_size,img_size) y genera NHWC uint8 con batch=1.
        """
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(img_resized.astype(np.uint8), axis=0)  # [1,H,W,3]

    @staticmethod
    def _nms_np(boxes_xyxy, scores, iou_thresh=0.45):
        """NMS class-agnostic en NumPy (rápido y sin dependencias extra)."""
        if boxes_xyxy.shape[0] == 0:
            return []

        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]

        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h

            union = areas[i] + areas[order[1:]] - inter + 1e-9
            iou = inter / union

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs):
        """
        Convierte la salida de la NPU en una lista de detecciones:
        [{'class_id','class_name','confidence','bbox_xyxy'}] en coordenadas 0..img_size.
        Aplica:
          - sigmoid si hace falta
          - filtro por confianza (self.conf_th)
          - filtro de área mínima (self.min_box_frac)
          - NMS (self.iou_th), class-agnostic
        """
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]  # (N, 5+num_classes)
        pred = pred.astype(np.float32)

        xywh = pred[:, :4]
        obj = pred[:, 4]
        cls = pred[:, 5:]

        # Sigmoid si hace falta
        if np.nanmax(obj) > 1.0 or np.nanmin(obj) < 0.0:
            obj = 1.0 / (1.0 + np.exp(-obj))
        if np.nanmax(cls) > 1.0 or np.nanmin(cls) < 0.0:
            cls = 1.0 / (1.0 + np.exp(-cls))

        # Clase top y score final
        cls_ids = np.argmax(cls, axis=1)
        cls_conf = cls[np.arange(cls.shape[0]), cls_ids]
        scores = obj * cls_conf

        # Filtro confianza
        keep = scores >= self.conf_th
        if not np.any(keep):
            return []

        xywh = xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # xywh -> xyxy
        x_c, y_c, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # filtro por área mínima
        min_area = (self.img_size * self.img_size) * self.min_box_frac
        areas = (w * h)
        big = areas >= min_area
        if not np.any(big):
            return []

        boxes = boxes[big]
        scores = scores[big]
        cls_ids = cls_ids[big]

        # clip
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.img_size - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.img_size - 1)

        # top-k previo a NMS (velocidad)
        if boxes.shape[0] > self.nms_topk:
            top_idx = np.argsort(-scores)[: self.nms_topk]
            boxes = boxes[top_idx]
            scores = scores[top_idx]
            cls_ids = cls_ids[top_idx]

        # NMS class-agnostic
        keep_idx = self._nms_np(boxes, scores, iou_thresh=self.iou_th)
        if not keep_idx:
            return []

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_ids = cls_ids[keep_idx]

        # salida
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
        """pre → inferencia → post"""
        img_input = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self.postprocess(outputs)


# Prueba rápida (solo cuando se ejecute directamente este archivo)
if __name__ == "__main__":
    model = RknnModel()
    img = cv2.imread("imagen_de_prueba.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = model.predict(img)
    print("Detecciones:", detections)
