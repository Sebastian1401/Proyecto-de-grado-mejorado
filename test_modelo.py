import torch
import cv2
import yaml

# --- CONFIGURACIÓN ---
WEIGHTS_PATH = 'weights/model1.pt'
IMAGE_PATH = 'imagen_de_prueba.jpg'
DATA_YAML_PATH = 'data/data.yaml' # Archivo con los nombres de las clases

print("--- Iniciando prueba del modelo YOLOv5 ---")

# 1. Cargar el modelo
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS_PATH, trust_repo=True)
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ ERROR al cargar el modelo: {e}")
    exit()

# 2. Cargar la imagen
try:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ ERROR: No se pudo cargar la imagen desde '{IMAGE_PATH}'.")
        exit()
    print("✅ Imagen de prueba cargada.")
except Exception as e:
    print(f"❌ ERROR al leer la imagen: {e}")
    exit()
    
# 3. Cargar nombres de las clases
try:
    with open(DATA_YAML_PATH, 'r') as stream:
        class_names = yaml.safe_load(stream)['names']
    print(f"✅ Nombres de clases cargados: {class_names}")
except Exception as e:
    print(f"❌ ERROR al cargar '{DATA_YAML_PATH}': {e}")
    exit()

# 4. Realizar la predicción
print("\n--- Realizando predicción... ---")
results = model(img)

# 5. Mostrar los resultados
detections = results.pred[0]
print(f"Se encontraron {len(detections)} detecciones.")

if len(detections) > 0:
    print("\n--- DETALLES DE LAS DETECCIONES ---")
    for i, (*xyxy, conf, cls) in enumerate(detections):
        class_name = class_names[int(cls)]
        print(f"  Detección #{i+1}: Clase='{class_name}', Confianza={conf:.2f}")
else:
    print("El modelo no encontró objetos en la imagen con la confianza suficiente.")

print("\n--- Prueba finalizada ---")