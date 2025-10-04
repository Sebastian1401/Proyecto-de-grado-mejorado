import cv2
import yaml
import json
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite

# --- CONFIG ---
RKNN_MODEL = './weights/model1.rknn'
IMAGE_PATH = './imagen_de_prueba.jpg'
DATA_YAML_PATH = './data/data.yaml'
IMG_SIZE = 640

# Perillas para suprimir cajas repetidas/ruido
CONF_TH = 0.60          # sube/baja 0.50~0.70
IOU_TH  = 0.30          # baja 0.25 si aún hay duplicadas
MIN_BOX_FRAC = 0.003    # 0.3% del área (sube si hay cajitas diminutas)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def postprocess(outputs, img_size=640, conf_threshold=0.6, iou_threshold=0.30, min_box_frac=0.003, nms_topk=300):
    """
    outputs: [ndarray] con shape (1,25200,20) o (25200,20)
    columnas: [cx, cy, w, h, obj, c0..c14]
    devuelve: boxes_xyxy (N,4), scores (N,), cls_ids (N,)
    """
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]  # (25200, 20)
    pred = pred.astype(np.float32)

    # separar
    xywh = pred[:, 0:4].copy()
    obj  = pred[:, 4].copy()
    cls  = pred[:, 5:].copy()

    # normalizar si hiciera falta
    if np.nanmax(obj) > 1.0 or np.nanmin(obj) < 0.0:
        obj = sigmoid(obj)
    if np.nanmax(cls) > 1.0 or np.nanmin(cls) < 0.0:
        cls = sigmoid(cls)

    # escala: si está en [0..1] pásalo a pixeles; si ya está en pixeles, déjalo
    if np.max(xywh) <= 1.5:
        xywh *= float(img_size)

    # clase top y score SIEMPRE con obj
    cls_ids  = np.argmax(cls, axis=1)
    cls_conf = cls[np.arange(cls.shape[0]), cls_ids]
    scores   = obj * cls_conf

    # filtrar por confianza
    keep = scores >= conf_threshold
    if not np.any(keep):
        return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))
    xywh, scores, cls_ids = xywh[keep], scores[keep], cls_ids[keep]

    # descartar cajas diminutas
    w, h = xywh[:,2], xywh[:,3]
    areas = w * h
    min_area = (img_size * img_size) * float(min_box_frac)
    big = areas >= min_area
    if not np.any(big):
        return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))
    xywh, scores, cls_ids = xywh[big], scores[big], cls_ids[big]

    # xywh -> xyxy y clip
    x_c, y_c, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    x1 = x_c - w/2; y1 = y_c - h/2
    x2 = x_c + w/2; y2 = y_c + h/2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy[:, [0,2]] = np.clip(boxes_xyxy[:, [0,2]], 0, img_size-1)
    boxes_xyxy[:, [1,3]] = np.clip(boxes_xyxy[:, [1,3]], 0, img_size-1)

    # NMS (class-agnostic)
    if len(scores) == 0:
        return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))

    # top-k previo a NMS para velocidad
    if len(scores) > nms_topk:
        top_idx = np.argsort(-scores)[:nms_topk]
        boxes_xyxy = boxes_xyxy[top_idx]
        scores     = scores[top_idx]
        cls_ids    = cls_ids[top_idx]

    # OpenCV NMSBoxes usa [x,y,w,h]
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:,2:4] = boxes_xywh[:,2:4] - boxes_xywh[:,0:2]
    boxes_cv = boxes_xywh.astype(np.int32).tolist()
    scores_cv = scores.astype(float).tolist()

    idxs = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, conf_threshold, iou_threshold)
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)) and len(idxs) and isinstance(idxs[0], (list, tuple)):
        idxs = [i[0] for i in idxs]
    if not idxs:
        return (np.empty((0,4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))

    return boxes_xyxy[idxs], scores[idxs], cls_ids[idxs]


if __name__ == '__main__':
    print("--- Iniciando prueba del modelo en NPU ---")

    # 1) clases
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            class_names = yaml.safe_load(f)['names']
        print(f"✅ Nombres de clases: {len(class_names)}")
    except Exception as e:
        print(f"❌ ERROR al cargar '{DATA_YAML_PATH}': {e}")
        exit(1)

    # 2) imagen
    try:
        img_orig = cv2.imread(IMAGE_PATH)
        if img_orig is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en {IMAGE_PATH}")
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_input = np.expand_dims(img.astype(np.uint8), axis=0)  # [1,H,W,3]
        print("✅ Imagen de prueba lista.")
    except Exception as e:
        print(f"❌ ERROR al cargar imagen: {e}")
        exit(1)

    # 3) RKNNLite
    rknn = RKNNLite(verbose=True)
    print(f"--> Cargando modelo: {RKNN_MODEL}")
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print("❌ ERROR al cargar .rknn"); rknn.release(); exit(1)
    print("--> Inicializando runtime...")
    if rknn.init_runtime() != 0:
        print("❌ ERROR al iniciar runtime"); rknn.release(); exit(1)
    print("✅ NPU listo.")

    # 4) inferencia
    outputs = rknn.inference(inputs=[img_input])
    print("[DEBUG] salida:", type(outputs), len(outputs), "shape0=", outputs[0].shape, "dtype=", outputs[0].dtype)

    # 5) post-proceso
    boxes, confs, clss = postprocess(
        outputs,
        img_size=IMG_SIZE,
        conf_threshold=CONF_TH,
        iou_threshold=IOU_TH,
        min_box_frac=MIN_BOX_FRAC
    )
    print(f"Se encontraron {len(boxes)} detecciones.")

    # 6) guardar y anotar
    out_dir = Path("runs_npu"); out_dir.mkdir(parents=True, exist_ok=True)

    det_list = []
    for b, s, c in zip(boxes, confs, clss):
        det_list.append({
            "class_id": int(c),
            "class_name": class_names[int(c)],
            "confidence": float(s),
            "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        })

    det_list.sort(key=lambda d: d["confidence"], reverse=True)
    print("\n=== DETECCIONES (top 20) ===")
    for d in det_list[:20]:
        print(f'{d["class_name"]}: {d["confidence"]:.3f}  bbox={d["bbox_xyxy"]}')

    with open(out_dir / "detecciones.json", "w", encoding="utf-8") as f:
        json.dump(det_list, f, ensure_ascii=False, indent=2)
    with open(out_dir / "detecciones.txt", "w", encoding="utf-8") as f:
        for d in det_list:
            f.write(f'{d["class_name"]} {d["confidence"]:.4f} ' + " ".join(f"{v:.1f}" for v in d["bbox_xyxy"]) + "\n")
    print(f"\n✅ Resultados guardados en: {out_dir/'detecciones.json'} y {out_dir/'detecciones.txt'}")

    # anotar sobre la resolución original
    H, W = img_orig.shape[:2]
    scale_x = W / float(IMG_SIZE); scale_y = H / float(IMG_SIZE)
    img_anno = img_orig.copy()
    for b, s, c in zip(boxes, confs, clss):
        x1, y1, x2, y2 = b
        x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
        cv2.rectangle(img_anno, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_names[int(c)]} {float(s):.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - 5)
        cv2.rectangle(img_anno, (x1, y_text - th - 4), (x1 + tw + 2, y_text), (0, 255, 0), -1)
        cv2.putText(img_anno, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    img_out_path = out_dir / "salida_annotada.jpg"
    cv2.imwrite(str(img_out_path), img_anno)
    print(f"🖼️ Imagen anotada guardada en: {img_out_path}")

    rknn.release()
    print("\n--- Prueba finalizada ---")
