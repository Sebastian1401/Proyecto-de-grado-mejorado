"""Blueprint: cámara/stream/predicción/captura."""

import os
import io
import time
import cv2
import numpy as np
from datetime import datetime
from flask import Blueprint, Response, request, jsonify
from app.services.inference_service import InferenceService
from app.services.patient_service import PatientService
_patients = PatientService()

# Globals controlados por este módulo (estado de cámara/stream)
bp = Blueprint("camera", __name__)
_camera = None
_predictions_enabled = False
_current_frame = None
_last_boxes = []
_last_ts = 0.0
HOLD_MS = 250

# Servicio de inferencia (usa RKNN adapter)
_infer = InferenceService()


def _draw_detections(frame_bgr, dets, img_size=640):
    """Dibuja cajas y etiquetas, reescalando de 640x640 a resolución original."""
    H, W = frame_bgr.shape[:2]
    sx, sy = W / float(img_size), H / float(img_size)

    for d in dets:
        conf = float(d["confidence"])
        cls_name = d["class_name"]
        etiqueta = _infer.label_for_class(cls_name)
        conf = _infer.adjust_conf(conf)

        x1, y1, x2, y2 = d["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        texto = f"{etiqueta} {conf:.2f}"
        cv2.putText(
            frame_bgr,
            texto,
            (max(0, x2 - 150), max(0, y2 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return frame_bgr


def _stream_generator():
    """Genera frames JPEG para MJPEG stream."""
    global _camera, _current_frame, _predictions_enabled

    if _camera is None:
        _camera = cv2.VideoCapture(0)

    while True:
        ok, frame = _camera.read()
        if not ok:
            # Evita tight loop si la cámara cae
            time.sleep(0.05)
            continue

        if _predictions_enabled:
            dets = _infer.predict(frame)

            now = time.time() * 1000.0
            global _last_boxes, _last_ts
            if len(dets) == 0 and (now - _last_ts) < HOLD_MS:
                dets_to_draw = _last_boxes
            else:
                dets_to_draw = dets
                if len(dets) > 0:
                    _last_boxes = dets
                    _last_ts = now

            frame = _draw_detections(frame, dets_to_draw, img_size=_infer.img_size)


        _current_frame = frame

        # Codificar a JPEG y renderear chunk
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            # Si falla, intenta siguiente
            time.sleep(0.01)
            continue

        chunk = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        yield chunk
        # No saturar CPU; si usas gevent, esto puede ser gevent.sleep(0)
        time.sleep(0.01)


@bp.route("/video_feed")
def video_feed():
    """Endpoint del stream de cámara."""
    return Response(_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/toggle_predictions", methods=["POST"])
def toggle_predictions():
    """Activa/desactiva inferencia en caliente (UI switch)."""
    global _predictions_enabled
    enabled = request.form.get("enabled") == "true"
    _predictions_enabled = enabled
    return "OK"


@bp.route("/capture", methods=["POST"])
def capture():
    try:
        cedula = request.form["cedula"]
        image_file = request.files["image"]

        filestr = image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        frame_capturado = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame_capturado is None:
            return jsonify({"message": "Error: imagen inválida"}), 400

        filename = _patients.save_capture_blob(cedula, frame_capturado)
        return jsonify({"message": "Foto capturada correctamente", "filename": filename})
    except Exception as e:
        return jsonify({"message": f"Ocurrió un error en el servidor: {e}"}), 500