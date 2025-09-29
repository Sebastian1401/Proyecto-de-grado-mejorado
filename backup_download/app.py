import os
from flask import Flask, render_template, Response, request, send_file
import cv2
import torch
from datetime import datetime
import zipfile

app = Flask(__name__)

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/model1.pt', force_reload=True)
model.eval()

# Variable global para almacenar el último frame
current_frame = None
camera = None  # Variable global para la cámara
predictions_enabled = False  # Variable global para controlar el estado de las predicciones

# Función para realizar el videostreaming con detecciones
def generate_camera():
    global current_frame, camera, predictions_enabled
    camera = cv2.VideoCapture(1)  # Abre la cámara (ajusta el índice si es necesario)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Procesar el frame solo si las predicciones están habilitadas
        if predictions_enabled:
            # Convertir el frame a BGR y hacer predicción
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = model(frame_bgr)

            # Obtener las detecciones
            detections = results.pred[0]
            
            for *xyxy, conf, cls in detections:
                if conf > 0.5:
                    xyxy = [int(i) for i in xyxy]
                    # Dibujar las cajas en el frame
                    frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                    frame = cv2.putText(frame, f'Conf: {conf:.2f}', (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Guardar el último frame para capturarlo cuando el usuario lo requiera
        current_frame = frame

        # Codificar el frame como JPEG para el streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Enviar el frame como parte del stream de video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()  # Liberar la cámara al finalizar el streaming

# Ruta para el videostreaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para capturar y guardar la imagen sin interrumpir el streaming
@app.route('/capture', methods=['POST'])
def capture():
    global current_frame
    cedula = request.form['cedula']  # Obtener la cédula del formulario
    message = ""  # Mensaje de éxito

    if current_frame is not None:
        # Crear la carpeta del paciente si no existe
        patient_folder = os.path.join('resultados_prueba', cedula)
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

        # Guardar la imagen en la carpeta del paciente con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = os.path.join(patient_folder, f'captura_{timestamp}.jpg')

        # Guardar la imagen en el backend
        cv2.imwrite(img_filename, current_frame)
        message = "Foto capturada correctamente"

    return render_template('camera.html', cedula=cedula, message=message)

# Ruta principal para mostrar la interfaz de ingreso de datos del paciente
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
    with open(os.path.join(patient_folder, 'datos_paciente.txt'), 'a') as f:
        f.write(f'Nombre: {nombre}, Cédula: {cedula}, Edad: {edad}, Género: {genero}, Antecedentes: {antecedentes}\n')

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
    global predictions_enabled
    predictions_enabled = request.form.get('enabled') == 'true'  # Convertir el valor a booleano
    return "Estado de predicciones actualizado"

@app.route('/download/<cedula>', methods=['GET'])
def download_data(cedula):
    patient_folder = f'resultados_prueba/{cedula}'
    zip_filename = f'data_zip/{cedula}_data.zip'

    try:
        # Verificar si la carpeta del paciente existe
        if not os.path.exists(patient_folder):
            return "Error: La carpeta del paciente no existe.", 404

        # Crear un archivo ZIP
        if not os.path.exists('data_zip'):
            os.makedirs('data_zip')

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, dirs, files in os.walk(patient_folder):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), patient_folder))

        # Verificar si el archivo ZIP fue creado correctamente
        if not os.path.exists(zip_filename):
            return "Error: No se pudo crear el archivo ZIP.", 500

        # Enviar el archivo ZIP al cliente
        return send_file(zip_filename, as_attachment=True)

    except Exception as e:
        # Retornar el mensaje de error
        return f"Ocurrió un error: {str(e)}", 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)