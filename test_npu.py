import cv2
import yaml
import json
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite

# --- CONFIGURACIÓN ---
RKNN_MODEL = './weights/model1.rknn'
IMAGE_PATH = './imagen_de_prueba.jpg'
DATA_YAML_PATH = './data/data.yaml'
IMG_SIZE = 640

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def postprocess(outputs, img_size=640, conf_threshold=0.25, iou_threshold=0.45, nms_topk=300):
    """
    Espera outputs = [array] con shape (1,25200,20) o (25200,20)
    Layout esperado (más común): [x, y, w, h, obj, c0..c14] (20 cols)
    Se auto-ajusta si:
    - x,y,w,h ya están en píxeles (max>1.5) o en [0..1]
    - obj/clases requieren sigmoide (max>1) o ya están en [0..1]
    Devuelve: boxes_xyxy (N,4), scores (N,), cls_ids (N,)
    """
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]  # (25200, 20)
    pred = pred.astype(np.float32)

    # Debug rápido de columnas
    col_max = pred.max(axis=0)
    col_min = pred.min(axis=0)
    print(f"[PP] col0..3(xywh) min/max: {col_min[:4]} / {col_max[:4]}")
    print(f"[PP] col4(obj) min/max: {col_min[4]:.4f} / {col_max[4]:.4f}")
    print(f"[PP] col5..19(cls) max: {col_max[5:20]} (max_global={col_max[5:20].max():.4f})")

    xywh = pred[:, 0:4].copy()
    obj  = pred[:, 4].copy()
    cls  = pred[:, 5:].copy()  # (25200, 15)

    # ¿Las clases/obj requieren sigmoide?
    need_sigmoid_obj = np.nanmax(obj) > 1.0 or np.nanmin(obj) < 0.0
    need_sigmoid_cls = np.nanmax(cls) > 1.0 or np.nanmin(cls) < 0.0

    if need_sigmoid_obj:
        obj = sigmoid(obj)
    if need_sigmoid_cls:
        cls = sigmoid(cls)

    # En algunos exports NO hay obj real (la col4 es otra cosa o ~1 constante).
    # Si el percentil 95 de 'obj' es muy bajo (<0.2) pero las clases tienen valores decentes,
    # probamos puntuar solo con clases.
    use_obj = True
    if np.percentile(obj, 95) < 0.2 and np.percentile(cls.max(axis=1), 95) > 0.2:
        use_obj = False
        print("[PP] Aviso: 'obj' parece no ser usable; se usará solo confianza de clase.")

    # ¿xywh en píxeles o relativos?
    if np.max(xywh[:,:4]) > 1.5:  # parecen píxeles 0..640
        scale = 1.0
    else:
        scale = float(img_size)
        xywh *= scale

    # Mejor clase y score
    cls_ids  = np.argmax(cls, axis=1)                # (25200,)
    cls_conf = cls[np.arange(cls.shape[0]), cls_ids] # (25200,)
    scores = cls_conf if not use_obj else (obj * cls_conf)

    # Filtro por umbral (arranca bajo y sube luego si hace falta)
    keep = scores >= conf_threshold
    if not np.any(keep):
        # intenta bajar umbral si todo quedó en 0 porque obj era pequeño
        fallback = (cls_conf >= max(0.05, conf_threshold * 0.5))
        if np.any(fallback):
            print("[PP] Reintento: usando solo confianza de clase con umbral más bajo.")
            scores = cls_conf
            keep = fallback
        else:
            return (np.empty((0,4), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.int32))

    xywh = xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # xywh centro → xyxy
    x_c, y_c, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    x1 = x_c - w/2; y1 = y_c - h/2
    x2 = x_c + w/2; y2 = y_c + h/2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    # Clip
    boxes_xyxy[:, [0,2]] = np.clip(boxes_xyxy[:, [0,2]], 0, img_size-1)
    boxes_xyxy[:, [1,3]] = np.clip(boxes_xyxy[:, [1,3]], 0, img_size-1)

    # NMS (OpenCV)
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:,2:4] = boxes_xywh[:,2:4] - boxes_xywh[:,0:2]
    boxes_cv = boxes_xywh.astype(np.int32).tolist()
    scores_cv = scores.astype(float).tolist()

    # Limita top-K para acelerar NMS si hay muchas
    if len(scores_cv) > nms_topk:
        top_idx = np.argsort(-scores)[:nms_topk]
        boxes_cv = [boxes_cv[i] for i in top_idx]
        scores_cv = [scores_cv[i] for i in top_idx]
        boxes_xyxy = boxes_xyxy[top_idx]
        scores = scores[top_idx]
        cls_ids = cls_ids[top_idx]

    idxs = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, max(1e-6, conf_threshold*0.5), iou_threshold)
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)) and len(idxs) and isinstance(idxs[0], (list, tuple)):
        idxs = [i[0] for i in idxs]
    if not idxs:
        return (np.empty((0,4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32))

    return boxes_xyxy[idxs], scores[idxs], cls_ids[idxs]


if __name__ == '__main__':
    print("--- Iniciando prueba del modelo en NPU ---")

    # 1. Cargar nombres de las clases
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            class_names = yaml.safe_load(f)['names']
        print(f"✅ Nombres de clases cargados: {len(class_names)} clases")
    except Exception as e:
        print(f"❌ ERROR al cargar '{DATA_YAML_PATH}': {e}")
        exit(1)

    # 2. Cargar imagen y pre-procesarla
    try:
        img_orig = cv2.imread(IMAGE_PATH)
        if img_orig is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en {IMAGE_PATH}")
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # RKNN Lite normalmente espera NHWC uint8; añadimos batch
        img_input = img.astype(np.uint8)
        img_input = np.expand_dims(img_input, axis=0)  # [1, H, W, 3]

        print("✅ Imagen de prueba cargada y pre-procesada.")
    except Exception as e:
        print(f"❌ ERROR al cargar la imagen '{IMAGE_PATH}': {e}")
        exit(1)

    # 3. Iniciar el entorno del NPU con RKNNLite
    rknn = RKNNLite(verbose=True)
    print(f"--> Cargando modelo RKNN: {RKNN_MODEL}")
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("❌ ERROR: Fallo al cargar el modelo .rknn")
        rknn.release()
        exit(1)

    print("--> Inicializando el entorno de ejecución del NPU...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("❌ ERROR: Fallo al inicializar el entorno del NPU")
        rknn.release()
        exit(1)
    print("✅ Entorno del NPU listo.")

    # 4. Realizar la inferencia
    print("\n--- Realizando inferencia en el NPU... ---")
    outputs = rknn.inference(inputs=[img_input])
    print("[DEBUG] type(outputs):", type(outputs))
    print("[DEBUG] len(outputs):", len(outputs))
    print("[DEBUG] outputs[0].shape:", outputs[0].shape, "dtype:", outputs[0].dtype)

    pred = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]  # (25200,20)
    print("[DBG] per-column max:", pred.max(axis=0))
    print("[DBG] per-column min:", pred.min(axis=0))

    # Mira top filas por "mejor clase" ANTES de filtrado, con y sin sigmoide
    raw_cls = pred[:,5:]
    raw_obj = pred[:,4]
    best_cls_id = np.argmax(raw_cls, axis=1)
    best_cls_raw = raw_cls[np.arange(raw_cls.shape[0]), best_cls_id]
    best_cls_sig = 1/(1+np.exp(-raw_cls))
    best_cls_sig = best_cls_sig[np.arange(raw_cls.shape[0]), best_cls_id]
    obj_sig = 1/(1+np.exp(-raw_obj))

    top = np.argsort(-best_cls_sig)[:10]
    print("\n[DBG] Top-10 por clase(sigmoid):")
    for i in top:
        print(f"i={i:5d} obj_raw={raw_obj[i]:8.3f} obj_sig={obj_sig[i]:.3f} "
            f"cls_id={best_cls_id[i]:2d} cls_raw={best_cls_raw[i]:8.3f} cls_sig={best_cls_sig[i]:.3f} "
            f"xywh={pred[i,:4]}")


    # === 1) Obtener y revisar el tensor de salida ===
    # Si tu llamada actual es algo como: outputs = rknn_lite.inference(inputs=[img])
    pred = outputs[0]  # (1, 25200, 20) o (25200, 20)
    pred = np.squeeze(pred)  # -> (25200, 20)

    print(f"[DEBUG] output shape={pred.shape}, dtype={pred.dtype}, min={pred.min():.4f}, max={pred.max():.4f}")

    # Si el dtype fuese int8/uint8, intenta obtenerlo en float:
    # (Descomenta si tu API lo soporta; en muchas versiones de RKNN Lite2 existe get_outputs con out_type='float')
    # pred = rknn_lite.get_outputs(out_type='float')[0]
    # pred = np.squeeze(pred)

    # === 2) Separar boxes/obj/clases ===
    # Asumimos formato estándar: [cx, cy, w, h, obj, num_classes=15]
    num_classes = 15
    boxes   = pred[:, :4]
    obj     = pred[:, 4:5]
    cls_raw = pred[:, 5:5+num_classes]

    # Si la red ya aplicó Sigmoid dentro del grafo (suele ser así en RKNN), esto ya viene en [0,1].
    # Si ves valores fuera de [0,1], aplica: boxes[:, :2] = 1/(1+np.exp(-boxes[:, :2])) ... etc. (pero normalmente NO hace falta).

    # === 3) Convertir cx,cy,w,h -> x1,y1,x2,y2 (en pixeles del input 640x640) ===
    # Ajusta si tu preproceso usa letterbox/escala distinta:
    img_size = 640.0
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = (cx - w/2) * img_size
    y1 = (cy - h/2) * img_size
    x2 = (cx + w/2) * img_size
    y2 = (cy + h/2) * img_size
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # === 4) Puntuación por clase y clase top ===
    cls_prob = cls_raw  # normalmente ya está en [0,1]
    cls_id   = np.argmax(cls_prob, axis=1)
    cls_sc   = cls_prob[np.arange(cls_prob.shape[0]), cls_id]

    # Puntaje final típico: obj * score_de_clase
    scores = (obj[:,0] * cls_sc)

    # === 5) Filtrar por umbral y Non-Maximum Suppression (NMS) ===
    conf_th = 0.25
    iou_th  = 0.45

    mask = scores >= conf_th
    boxes_xyxy = boxes_xyxy[mask]
    scores     = scores[mask]
    cls_id     = cls_id[mask]

    def nms(boxes, scores, iou_thresh=0.45):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    keep = nms(boxes_xyxy, scores, iou_thresh=iou_th)

    boxes_xyxy = boxes_xyxy[keep]
    scores     = scores[keep]
    cls_id     = cls_id[keep]

    # === 6) Mapear IDs a nombres de clase y mostrar por consola ===
    # Asegúrate de que 'class_names' es tu lista de 15 nombres ya cargada
    det_list = []
    for b, s, c in zip(boxes_xyxy, scores, cls_id):
        det_list.append({
            "class_id": int(c),
            "class_name": class_names[int(c)],
            "confidence": float(s),
            "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        })

    # Ordenar por confianza desc y mostrar top 20
    det_list.sort(key=lambda d: d["confidence"], reverse=True)
    print("\n=== DETECCIONES (top 20) ===")
    for d in det_list[:20]:
        print(f'{d["class_name"]}: {d["confidence"]:.3f}  bbox={d["bbox_xyxy"]}')

    # === 7) Guardar a archivos (útil por SSH) ===
    out_dir = Path("runs_npu")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = out_dir / "detecciones.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(det_list, f, ensure_ascii=False, indent=2)

    # TXT simple (una línea por detección)
    txt_path = out_dir / "detecciones.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in det_list:
            f.write(f'{d["class_name"]} {d["confidence"]:.4f} ' +
                    " ".join(f"{v:.1f}" for v in d["bbox_xyxy"]) + "\n")

    print(f"\n✅ Resultados guardados en: {json_path} y {txt_path}")



    # 5. Post-procesar y mostrar/guardar resultados (con imagen anotada)
    

    boxes, confs, clss = postprocess(outputs, img_size=IMG_SIZE, conf_threshold=0.25, iou_threshold=0.45)
    print(f"Se encontraron {len(boxes)} detecciones.")

    # Crear carpeta de salida
    out_dir = Path("runs_npu")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Si hay detecciones, mostrar algunas por consola
    if len(boxes) > 0:
        print("\n--- DETALLES DE LAS DETECCIONES (top 20) ---")
        # ordenar por confianza desc
        order = np.argsort(-confs)
        order = order[:20]
        for i in order:
            class_name = class_names[int(clss[i])]
            print(f"  - Clase: '{class_name}', Confianza: {float(confs[i]):.3f}, "
                f"bbox_xyxy={[float(v) for v in boxes[i]]}")

    # Guardar TXT y JSON
    det_list = []
    for b, s, c in zip(boxes, confs, clss):
        det_list.append({
            "class_id": int(c),
            "class_name": class_names[int(c)],
            "confidence": float(s),
            "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        })

    json_path = out_dir / "detecciones.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(det_list, f, ensure_ascii=False, indent=2)

    txt_path = out_dir / "detecciones.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in det_list:
            f.write(f'{d["class_name"]} {d["confidence"]:.4f} ' +
                    " ".join(f"{v:.1f}" for v in d["bbox_xyxy"]) + "\n")

    print(f"\n✅ Resultados guardados en: {json_path} y {txt_path}")

    # 6. Guardar imagen anotada (no hay GUI, luego la descargas por scp)
    #   - Tu preproceso reescaló a 640x640, así que escalamos de vuelta a la resolución original.
    H, W = img_orig.shape[:2]
    scale_x = W / float(IMG_SIZE)
    scale_y = H / float(IMG_SIZE)

    img_anno = img_orig.copy()
    for b, s, c in zip(boxes, confs, clss):
        x1, y1, x2, y2 = b
        x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
        cv2.rectangle(img_anno, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_names[int(c)]} {float(s):.2f}'
        # Dibujar fondo del texto para legibilidad
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - 5)
        cv2.rectangle(img_anno, (x1, y_text - th - 4), (x1 + tw + 2, y_text), (0, 255, 0), -1)
        cv2.putText(img_anno, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    img_out_path = out_dir / "salida_annotada.jpg"
    cv2.imwrite(str(img_out_path), img_anno)
    print(f"🖼️ Imagen anotada guardada en: {img_out_path}")


    # 6. Liberar recursos
    rknn.release()
    print("\n--- Prueba finalizada ---")