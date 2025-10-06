"""Service: empaqueta ZIP con PDF e imágenes de un paciente."""
import io, os, zipfile
from typing import Tuple
from app.services.patient_service import PatientService
from app.adapters.pdf_reportlab import build_report

class ReportService:
    def __init__(self, patient_svc: PatientService | None = None) -> None:
        self.patient = patient_svc or PatientService()

    def make_zip_for_patient(self, cedula: str) -> Tuple[str, io.BytesIO]:
        info = self.patient.get_patient_info(cedula)
        imagenes = self.patient.list_captures(cedula)
        # generar PDF en la carpeta del paciente
        pdf_name = "informe_medico.pdf"
        pdf_path = self.patient.storage.file_path(cedula, pdf_name)
        img_paths = [self.patient.storage.file_path(cedula, f) for f in imagenes]
        build_report(pdf_path, info, img_paths)

        # zip en memoria
        mem = io.BytesIO()
        pdir = self.patient.storage.patient_dir(cedula)
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(pdir):
                zf.write(os.path.join(pdir, fname), fname)
        mem.seek(0)
        zip_filename = f"{cedula}_resultados.zip"
        return zip_filename, mem