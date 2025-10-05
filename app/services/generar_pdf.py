import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime

def generar_pdf(pdf_path, nombre, cedula, edad, genero, antecedentes, imagenes_paths):
    try:
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        story = []

        # Título del informe
        story.append(Paragraph("Informe Médico de Paciente", styles['h1']))
        story.append(Spacer(1, 0.2*inch))

        # Fecha del reporte
        now = datetime.now()
        fecha_reporte = now.strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Fecha del Reporte: {fecha_reporte}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Datos del Paciente
        story.append(Paragraph("<b>Datos del Paciente</b>", styles['h2']))
        story.append(Paragraph(f"<b>Nombre:</b> {nombre}", styles['Normal']))
        story.append(Paragraph(f"<b>Cédula:</b> {cedula}", styles['Normal']))
        story.append(Paragraph(f"<b>Edad:</b> {edad}", styles['Normal']))
        story.append(Paragraph(f"<b>Género:</b> {genero}", styles['Normal']))
        story.append(Paragraph("<b>Antecedentes:</b>", styles['Normal']))
        story.append(Paragraph(antecedentes, styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))

        # Imágenes Capturadas
        if imagenes_paths:
            story.append(Paragraph("<b>Imágenes Capturadas</b>", styles['h2']))
            for img_path in imagenes_paths:
                try:
                    # Ajusta el tamaño de la imagen para que quepa en la página
                    img = Image(img_path, width=4*inch, height=3*inch, kind='proportional')
                    story.append(img)
                    story.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    print(f"Error al añadir la imagen {img_path} al PDF: {e}")
                    story.append(Paragraph(f"<i>Error al cargar imagen: {os.path.basename(img_path)}</i>", styles['Italic']))
        
        doc.build(story)
        print(f"PDF generado exitosamente en {pdf_path}")
    
    except Exception as e:
        print(f"ERROR FATAL al generar PDF: {e}")