from flask import Blueprint, render_template, request, jsonify, send_file
from app.services.patient_service import PatientService
from app.services.report_service import ReportService

bp = Blueprint("pages", __name__)
_patients = PatientService()
_reports = ReportService(_patients)

@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/start_stream", methods=["POST"])
def start_stream():
    nombre = request.form["nombre"]
    cedula = request.form["cedula"]
    edad = request.form["edad"]
    genero = request.form["genero"]
    antecedentes = request.form["antecedentes"]

    _patients.save_patient_info({
        "Nombre": nombre,
        "Cédula": cedula,
        "Edad": edad,
        "Género": genero,
        "Antecedentes": antecedentes,
    })
    return render_template("camera.html", cedula=cedula)

@bp.route("/stop_stream", methods=["POST"])
def stop_stream():
    # la cámara se gestiona en camera.py; aquí no hay nada que soltar
    return "Stream detenido"

@bp.route("/download/<cedula>")
def download_data(cedula):
    zip_filename, mem = _reports.make_zip_for_patient(cedula)
    return send_file(mem, download_name=zip_filename, as_attachment=True)

@bp.route("/historial")
def historial():
    pacientes = _patients.list_patients_summary()
    return render_template("historial.html", pacientes=pacientes)