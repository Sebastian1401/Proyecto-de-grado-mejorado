import cv2
import numpy as np
import yaml
from rknn.api import RKNNLite

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

            # Filtra por confianza
            conf = o[..., 4]
            idx = conf > conf_threshold
            box, conf, o = box[idx], conf[idx], o[idx]

            # Obtiene la clase con mayor puntuación
            cls = np.argmax(o[..., 5:], axis=1)

            boxes.append(box)
            confs.append(conf)
            clss.append(cls)

    boxes = np.concatenate(boxes)
    confs = np.concatenate(confs)
    clss = np.concatenate(clss)

    # Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), conf_threshold, iou_threshold)

    return boxes[indices], confs[indices], clss[indices]

if __name__ == '__main__':
    print("--- Iniciando prueba del modelo en NPU ---")

    # 1. Cargar nombres de las clases
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            class_names = yaml.safe_load(f)['names']
        print(f"✅ Nombres de clases cargados: {len(class_names)} clases")
    except Exception as e:
        print(f"❌ ERROR al cargar '{DATA_YAML_PATH}': {e}")
        exit()

    # 2. Cargar imagen y pre-procesarla
    try:
        img_orig = cv2.imread(IMAGE_PATH)
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        print("✅ Imagen de prueba cargada y pre-procesada.")
    except Exception as e:
        print(f"❌ ERROR al cargar la imagen '{IMAGE_PATH}': {e}")
        exit()

    # 3. Iniciar el entorno del NPU
    rknn_lite = RKNNLite(verbose=False)
    print(f"--> Cargando modelo RKNN: {RKNN_MODEL}")
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("❌ ERROR: Fallo al cargar el modelo .rknn")
        exit()

    print("--> Inicializando el entorno de ejecución del NPU...")
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print("❌ ERROR: Fallo al inicializar el entorno del NPU")
        exit()
    print("✅ Entorno del NPU listo.")

    # 4. Realizar la inferencia
    print("\n--- Realizando inferencia en el NPU... ---")
    outputs = rknn_lite.inference(inputs=[img])

    # 5. Post-procesar y mostrar los resultados
    boxes, confs, clss = postprocess(outputs)
    print(f"Se encontraron {len(boxes)} detecciones.")

    if len(boxes) > 0:
        print("\n--- DETALLES DE LAS DETECCIONES ---")
        for box, conf, cls in zip(boxes, confs, clss):
            class_name = class_names[int(cls)]
            print(f"  - Clase: '{class_name}', Confianza: {conf:.2f}")

    # 6. Liberar recursos
    rknn_lite.release()
    print("\n--- Prueba finalizada ---")