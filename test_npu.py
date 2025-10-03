import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite   # <-- CAMBIO: usar Lite

# --- CONFIGURACIÓN ---
RKNN_MODEL = './weights/model1.rknn'
IMAGE_PATH = './imagen_de_prueba.jpg'
DATA_YAML_PATH = './data/data.yaml'
IMG_SIZE = 640

def postprocess(outputs, img_size=640, conf_threshold=0.25, iou_threshold=0.45):
    """
    Espera outputs = [array] donde array tiene shape (1,25200,20) o (25200,20)
    Formato columnas: [x, y, w, h, obj_conf, num_classes=15]
    Coord en XYWH relativas a [0..1] (común en YOLO exportado a RKNN).
    Devuelve: boxes_xyxy (N,4), scores (N,), cls_ids (N,)
    """
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]             # (25200, 20)
    pred = pred.astype(np.float32)

    # Separar campos
    xywh = pred[:, 0:4]           # (25200,4)
    obj  = pred[:, 4:5]           # (25200,1)
    cls  = pred[:, 5:]            # (25200,15)

    # Mejor clase por fila
    cls_ids  = np.argmax(cls, axis=1)                  # (25200,)
    cls_conf = cls[np.arange(cls.shape[0]), cls_ids]   # (25200,)

    # Confianza final (obj * clase)
    scores = (obj[:, 0] * cls_conf)

    # Filtro por umbral
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    xywh = xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # Pasar de XYWH centro → XYXY absolutos (en píxeles)
    x_c, y_c, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    # La mayoría de exports RKNN ya están en [0..1]; escalar:
    x_c *= img_size
    y_c *= img_size
    w   *= img_size
    h   *= img_size

    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Clip a la imagen
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_size - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_size - 1)

    # OpenCV NMSBoxes usa [x,y,w,h] int y lista de floats
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2:4] = boxes_xywh[:, 2:4] - boxes_xywh[:, 0:2]  # convertir a w,h
    boxes_cv = boxes_xywh.astype(np.int32).tolist()
    scores_cv = scores.astype(float).tolist()

    idxs = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, conf_threshold, iou_threshold)
    if len(idxs) == 0:
        return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # idxs puede ser array o lista de listas
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten()
    else:
        idxs = [i[0] if isinstance(i, (list, tuple)) else i for i in idxs]

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


    # 5. Post-procesar y mostrar los resultados
    boxes, confs, clss = postprocess(outputs, img_size=IMG_SIZE, conf_threshold=0.25, iou_threshold=0.45)
    print(f"Se encontraron {len(boxes)} detecciones.")

    if len(boxes) > 0:
        print("\n--- DETALLES DE LAS DETECCIONES ---")
        for box, conf, cls in zip(boxes, confs, clss):
            class_name = class_names[int(cls)]
            print(f"  - Clase: '{class_name}', Confianza: {float(conf):.2f}")

    # 6. Liberar recursos
    rknn.release()
    print("\n--- Prueba finalizada ---")