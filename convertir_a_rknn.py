from rknn.api import RKNN

# --- CONFIGURACIÓN ---
ONNX_MODEL = './weights/model1.onnx'  # Asegúrate que el nombre coincida con tu archivo ONNX
RKNN_MODEL = './weights/model1.rknn'
DATASET_FILE = './dataset.txt'

if __name__ == '__main__':
    # 1. Crear el objeto RKNN
    rknn = RKNN(verbose=True)

    # 2. Configurar el modelo
    print('--> Configurando el modelo...')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588'  # Plataforma para la Orange Pi 5
    )
    print('Configuración completa.')

    # 3. Cargar el modelo ONNX
    print(f'--> Cargando el modelo ONNX: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Error: Fallo al cargar el modelo ONNX.')
        exit(ret)
    print('Modelo ONNX cargado.')

    # 4. Construir el modelo RKNN (Este es el paso que tarda)
    print('--> Construyendo el modelo RKNN... Esto puede tardar varios minutos.')
    ret = rknn.build(do_quantization=True, dataset=DATASET_FILE)
    if ret != 0:
        print('Error: Fallo al construir el modelo RKNN.')
        exit(ret)
    print('Construcción completa.')

    # 5. Exportar el modelo RKNN
    print(f'--> Exportando a {RKNN_MODEL}')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Error: Fallo al exportar el modelo RKNN.')
        exit(ret)
    print(f'¡Éxito! Modelo guardado en {RKNN_MODEL}')

    rknn.release()
