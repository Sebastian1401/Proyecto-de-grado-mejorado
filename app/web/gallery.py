import os
from flask import Blueprint, jsonify, request, send_file, abort
from app.services.patient_service import PatientService

bp = Blueprint("gallery", __name__)
_patients = PatientService()

@bp.route("/get_capturas/<cedula>")
def get_capturas(cedula):
    try:
        files = _patients.list_captures(cedula)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/delete_captura", methods=["POST"])
def delete_captura():
    try:
        data = request.get_json()
        filename = data["filename"]
        cedula = data["cedula"]
        ok = _patients.delete_capture(cedula, filename)
        if ok:
            return jsonify({"success": True, "message": "Imagen eliminada"})
        return jsonify({"success": False, "message": "La imagen no existe"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# servir archivo (tal como usa camera.html: /resultados_prueba/<cedula>/<filename>)
@bp.route("/patients/<cedula>/<path:filename>")
def serve_capture(cedula, filename):
    base = os.path.abspath(_patients.storage.base_dir)
    filepath = os.path.join(base, cedula, filename)
    if not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath)