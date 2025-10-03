import cv2
import numpy as np
import yaml
from rknnlite.api import RKNNLite   # <-- CAMBIO: usar Lite

# --- CONFIGURACIÓN ---
RKNN_MODEL = './weights/model1.rknn'
IMAGE_PATH = './imagen_de_prueba.jpg'
DATA_YAML_PATH = './data/data.yaml'
IMG_SIZE = 640

def postprocess(outputs, conf_threshold=0.25, iou_threshold=0.45):
    # Lógica para decodificar la salida del modelo YOLOv5
    boxes, confs, clss = [], [], []
    for out in outputs:
        out = out.reshape([3, -1, 85])
        for o in out:
            o[..., 2:4] = (o[..., 2:4] ** 2) * 4
            box = o[..., :4]
            box[..., 0] *= IMG_SIZE
            box[..., 1] *= IMG_SIZE
            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2

            conf = o[..., 4]
            idx = conf > conf_threshold
            box, conf, o = box[idx], conf[idx], o[idx]

            cls = np.argmax(o[..., 5:], axis=1)

            boxes.append(box)
            confs.append(conf)
            clss.append(cls)

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes = np.concatenate(boxes)
    confs = np.concatenate(confs)
    clss = np.concatenate(clss)

    # cv2.dnn.NMSBoxes espera [x, y, w, h] y valores int
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2:4] = boxes_xywh[:, 2:4]  # ya son w,h
    boxes_int = boxes_xywh.astype(np.int32).tolist()
    confs_list = confs.astype(float).tolist()

    indices = cv2.dnn.NMSBoxes(boxes_int, confs_list, conf_threshold, iou_threshold)
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # indices puede ser lista de listas; aplanamos
    if isinstance(indices, np.ndarray):
        idxs = indices.flatten()
    else:
        idxs = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

    return boxes[idxs], confs[idxs], clss[idxs]

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

    # 5. Post-procesar y mostrar los resultados
    boxes, confs, clss = postprocess(outputs)
    print(f"Se encontraron {len(boxes)} detecciones.")

    if len(boxes) > 0:
        print("\n--- DETALLES DE LAS DETECCIONES ---")
        for box, conf, cls in zip(boxes, confs, clss):
            class_name = class_names[int(cls)]
            print(f"  - Clase: '{class_name}', Confianza: {float(conf):.2f}")

    # 6. Liberar recursos
    rknn.release()
    print("\n--- Prueba finalizada ---")