"""Adapter: ReportLab para generar PDF simple del paciente + imágenes."""
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def build_report(out_path: str, datos: dict, imagenes_paths: list[str]) -> None:
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("<b>Informe Médico</b>", styles["Title"]))
    elems.append(Spacer(1, 12))

    campos = ["Nombre", "Cédula", "Edad", "Género", "Antecedentes"]
    for c in campos:
        v = datos.get(c, "N/A")
        elems.append(Paragraph(f"<b>{c}:</b> {v}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    if imagenes_paths:
        elems.append(Paragraph("<b>Imágenes capturadas</b>", styles["Heading2"]))
        elems.append(Spacer(1, 8))
        for path in imagenes_paths:
            try:
                elems.append(Image(path, width=400, height=300))
                elems.append(Spacer(1, 8))
                elems.append(Paragraph(path.split("/")[-1], styles["Italic"]))
                elems.append(Spacer(1, 12))
            except Exception:
                continue

    doc.build(elems)