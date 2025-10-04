import gevent
from gevent import monkey
monkey.patch_all()

import os
from flask import Flask, render_template, Response, request, send_file, jsonify, send_from_directory
import cv2
from datetime import datetime
import zipfile
import io
import numpy as np
from utils.rknn_infer import RknnModel
from flask import jsonify
from generar_pdf import generar_pdf
from historial_handler import obtener_datos_pacientes


app = Flask(__name__)

# Cargar el modelo RKNN
RKNN_IMG_SIZE = 640
rknn_model = RknnModel(
    model_path='weights/model1.rknn',
    yaml_path='data/data.yaml',
    img_size=RKNN_IMG_SIZE
)


# Variable global para almacenar el último frame
current_frame = None
camera = None  # Variable global para la cámara
predictions_enabled = False  # Variable global para controlar el estado de las predicciones

# Función para realizar el videostreaming con detecciones
def generate_camera():
    global current_frame, camera, predictions_enabled
    camera = cv2.VideoCapture(0)  # Abre la cámara (ajusta el índice si es necesario)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Procesar el frame solo si las predicciones están habilitadas
        if predictions_enabled:
            H, W = frame.shape[:2]

            # RKNN: el wrapper espera RGB; internamente reescala a 640
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = rknn_model.predict(img_rgb)  # [{'class_id','class_name','confidence','bbox_xyxy'}] en 640x640

            # Reescalar cajas a la resolución original
            sx, sy = W / float(RKNN_IMG_SIZE), H / float(RKNN_IMG_SIZE)

            if len(dets) > 0:
                print(f"DEBUG: Modelo RKNN encontró {len(dets)} objetos ANTES del filtro de confianza.")

            for d in dets:
                conf = float(d["confidence"])
                cls_name = d["class_name"]

                # Grupos/etiqueta final (misma lógica que tuviste)
                grupos = {
                    'MALIGNO/PREMALIGNO': ['AKIEC', 'BCC', 'SCC', 'MEL'],
                    'BENIGNO': ['BKL', 'DF', 'NV', 'VASC']
                }
                if cls_name in grupos['MALIGNO/PREMALIGNO']:
                    etiqueta_final = "MALIGNO"
                elif cls_name in grupos['BENIGNO']:
                    etiqueta_final = "BENIGNO"
                else:
                    etiqueta_final = cls_name  # fallback

                # Ajuste de confianza (conservando tu heurística)
                if conf < 0.3:
                    conf += 0.3
                elif conf < 0.4:
                    conf += 0.2
                elif conf < 0.5:
                    conf += 0.1

                # Dibujar caja y texto
                x1, y1, x2, y2 = d["bbox_xyxy"]
                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                texto_a_mostrar = f'{etiqueta_final} {conf:.2f}'
                cv2.putText(frame, texto_a_mostrar,
                            (max(0, x2 - 150), max(0, y2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        # Guardar el último frame para capturarlo cuando el usuario lo requiera
        current_frame = frame

        # Codificar el frame como JPEG para el streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Enviar el frame como parte del stream de video
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        gevent.sleep(0)

    camera.release()  # Liberar la cámara al finalizar el streaming

# Ruta para el videostreaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    message = "Error: No se recibió ninguna imagen."
    try:
        cedula = request.form['cedula']
        image_file = request.files['image']
        
        filestr = image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        frame_capturado = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame_capturado is not None:
            patient_folder = os.path.join('resultados_prueba', cedula)
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f'captura_{timestamp}.jpg'
            full_path = os.path.join(patient_folder, img_filename)

            cv2.imwrite(full_path, frame_capturado)
            message = "Foto capturada correctamente"
            
            # Devolvemos también el nombre del archivo
            return jsonify({'message': message, 'filename': img_filename})
    
    except Exception as e:
        message = f"Ocurrió un error en el servidor: {e}"
        return jsonify({'message': message}), 500

    return jsonify({'message': message}), 400

@app.route('/get_capturas/<cedula>')
def get_capturas(cedula):
    patient_folder = os.path.join('resultados_prueba', cedula)
    if not os.path.isdir(patient_folder):
        return jsonify([]) # Devuelve una lista vacía si no existe la carpeta
    
    try:
        # Filtra solo los archivos .jpg y los ordena
        files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.jpg')])
        return jsonify(files)
    except Exception as e:
        print(f"ERROR en /get_capturas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_captura', methods=['POST'])
def delete_captura():
    try:
        data = request.get_json()
        filename = data['filename']
        cedula = data['cedula']

        # Medida de seguridad: asegurarse de que no se pueda borrar cualquier archivo
        if '..' in filename or filename.startswith('/'):
            return jsonify({'success': False, 'message': 'Nombre de archivo no válido'}), 400

        patient_folder = os.path.join('resultados_prueba', cedula)
        file_path = os.path.join(patient_folder, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': 'Imagen eliminada'})
        else:
            return jsonify({'success': False, 'message': 'La imagen no existe'}), 404
            
    except Exception as e:
        print(f"ERROR en /delete_captura: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    nombre = request.form['nombre']
    cedula = request.form['cedula']
    edad = request.form['edad']
    genero = request.form['genero']
    antecedentes = request.form['antecedentes']

    # Crear la carpeta del paciente si no existe
    patient_folder = os.path.join('resultados_prueba', cedula)
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)  # Crear carpeta si no existe

    # Guardar los datos en un archivo de texto dentro de la carpeta del paciente
    with open(os.path.join(patient_folder, 'datos_paciente.txt'), 'w') as f:
        f.write(f"Nombre: {nombre}\n")
        f.write(f"Cédula: {cedula}\n")
        f.write(f"Edad: {edad}\n")
        f.write(f"Género: {genero}\n")
        f.write(f"Antecedentes: {antecedentes}\n")

    return render_template('camera.html', cedula=cedula)

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global camera
    if camera is not None:
        camera.release()  # Liberar la cámara
    return "Stream detenido"

# Nueva ruta para habilitar/deshabilitar las predicciones
@app.route('/toggle_predictions', methods=['POST'])
def toggle_predictions():
    print("DEBUG: ¡La función /toggle_predictions ha sido llamada!")
    global predictions_enabled
    predictions_enabled = request.form.get('enabled') == 'true'
    print(f"DEBUG: El estado de 'predictions_enabled' ahora es: {predictions_enabled}")
    return "Estado de predicciones actualizado"

# === Perillas RKNN: get/set ===
@app.route('/thresholds', methods=['GET'])
def get_thresholds():
    """Devuelve los umbrales actuales del postproceso RKNN."""
    return jsonify(rknn_model.get_thresholds())

@app.route('/thresholds', methods=['POST'])
def set_thresholds():
    """
    Actualiza una o varias perillas:
    body JSON: { "conf_th": 0.6, "iou_th": 0.3, "min_box_frac": 0.003 }
        - conf_th e iou_th en [0,1]
        - min_box_frac en (0, 0.5]
    """
    data = request.get_json(force=True, silent=True) or {}

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    updates = {}
    if 'conf_th' in data:
        v = to_float(data['conf_th'])
        if v is None or not (0.0 <= v <= 1.0):
            return jsonify({"error": "conf_th debe estar en [0,1]"}), 400
        updates['conf_th'] = v

    if 'iou_th' in data:
        v = to_float(data['iou_th'])
        if v is None or not (0.0 <= v <= 1.0):
            return jsonify({"error": "iou_th debe estar en [0,1]"}), 400
        updates['iou_th'] = v

    if 'min_box_frac' in data:
        v = to_float(data['min_box_frac'])
        if v is None or not (0.0 < v <= 0.5):
            return jsonify({"error": "min_box_frac debe estar en (0, 0.5]"}), 400
        updates['min_box_frac'] = v

    if not updates:
        return jsonify({"error": "sin campos válidos (conf_th, iou_th, min_box_frac)"}), 400

    rknn_model.set_thresholds(**updates)
    return jsonify({"ok": True, "thresholds": rknn_model.get_thresholds()})

@app.route('/thresholds/reset', methods=['POST'])
def reset_thresholds():
    """Restaura valores por defecto usados en utils/rknn_infer.py"""
    rknn_model.set_thresholds(conf_th=0.60, iou_th=0.30, min_box_frac=0.003)
    return jsonify({"ok": True, "thresholds": rknn_model.get_thresholds()})


@app.route('/download/<cedula>', methods=['GET'])
def download_data(cedula):
    print(f"DEBUG: Iniciando descarga para cédula {cedula}")
    patient_folder = f'resultados_prueba/{cedula}'

    try:
        # Verificar si la carpeta del paciente existe
        if not os.path.exists(patient_folder):
            return "Error: La carpeta del paciente no existe.", 404

        # 1. Leer los datos del paciente desde el archivo de texto
        datos_paciente_path = os.path.join(patient_folder, 'datos_paciente.txt')
        datos_paciente = {}
        if os.path.exists(datos_paciente_path):
            with open(datos_paciente_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        datos_paciente[key.strip()] = value.strip()

        print("DEBUG: Datos del paciente leídos.")

        # 2. Recopilar las rutas de las imágenes capturadas
        imagenes_paths = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.jpg')]
        print(f"DEBUG: Se encontraron {len(imagenes_paths)} imágenes.")

        # 3. Definir la ruta del PDF y generarlo
        pdf_path = os.path.join(patient_folder, 'informe_medico.pdf')
        generar_pdf(
            pdf_path,
            datos_paciente.get('Nombre', 'N/A'),
            datos_paciente.get('Cédula', cedula),
            datos_paciente.get('Edad', 'N/A'),
            datos_paciente.get('Género', 'N/A'),
            datos_paciente.get('Antecedentes', 'N/A'),
            imagenes_paths
        )
        print("DEBUG: Función generar_pdf ejecutada.")

        # Crear un archivo ZIP en memoria
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_to_zip in os.listdir(patient_folder):
                zipf.write(os.path.join(patient_folder, file_to_zip), file_to_zip)

        memory_file.seek(0)
        zip_filename = f'{cedula}_resultados.zip'
        print(f"DEBUG: ZIP creado en memoria. Enviando archivo {zip_filename}")

        return send_file(memory_file, download_name=zip_filename, as_attachment=True)

    except Exception as e:
        print(f"ERROR FATAL en /download: {str(e)}")
        return f"Ocurrió un error al generar el ZIP: {str(e)}", 500

@app.route('/historial')
def historial():
    lista_de_pacientes = obtener_datos_pacientes()
    return render_template('historial.html', pacientes=lista_de_pacientes)

@app.route('/resultados_prueba/<path:filename>')
def serve_capture(filename):
    return send_from_directory('resultados_prueba', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
