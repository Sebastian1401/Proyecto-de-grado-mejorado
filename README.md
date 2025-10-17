<div align="center">

  <a href="github.com/Sebastian1401/Proyecto-de-grado-mejorado">
    <img 
      src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" 
      alt="Banner de NeuroDermascan"
      width="100%"
      style="max-width: 800px; border-radius: 10px;"
    >
  </a>

  <br/>

  <h1>NeuroDermaScan – Detección de lesiones cutáneas en Orange Pi 5</h1>

  <p>
    <strong>Una solución de IA en el borde para el análisis de lesiones cutáneas en tiempo real, potenciada por la NPU RK3588.</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Flask-2.0-black?logo=flask&logoColor=white" alt="Flask">
    <img src="https://img.shields.io/badge/OpenCV-4.5-green?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/Hardware-Orange%20Pi%205-orange" alt="Orange Pi 5">
    <img src="https://img.shields.io/badge/NPU-RKNN-red" alt="RKNN">
  </p>

</div>

## <div align="center">🎯 Objetivo</div>

Ofrecer un flujo completo “de laboratorio a clínica” que combine **desempeño en el borde (edge)**, **baja latencia** y **trazabilidad por paciente**, manteniendo los datos **en el dispositivo**.

## <div align="center">✨ Destacados</div>

-   🚀 **Inferencia acelerada por NPU:** Utiliza el máximo potencial del chip RK3588 para análisis en tiempo real.
-   🎥 **Streaming MJPEG estable:** Visualización en vivo fluida.
-   📂 **Gestión de pacientes:** Guarda y organiza capturas por cédula en la carpeta `var/patients/`.
-   🖥️ **UI optimizada:** Interfaz web simple con galería, descarga de datos y un historial para acceder a resultados anteriores.

## <div align="center">🛠️ Pila Técnica (Tech Stack)</div>

-   **Backend:** Python 3, Flask, Gunicorn (con gevent).
-   **Procesamiento de IA:** RKNN Toolkit / Adapter RKNN.
-   **Procesamiento de video:** OpenCV.
-   **Frontend:** HTML, CSS, JavaScript (con jQuery para peticiones AJAX).
-   **Hardware y SO:** Orange Pi 5 (Ubuntu/Debian-based) con drivers NPU.


## 🏗️ Arquitectura y Flujo de Trabajo

El sistema opera de manera local en la Orange Pi 5, siguiendo un flujo de datos claro y eficiente desde la captura hasta la visualización y el almacenamiento.

```mermaid
graph TD
    subgraph "Hardware Central: Orange Pi 5"
        A[🔬 Microscopio USB] -- Flujo de video (RAW) --> B{Backend Flask};
        B -- Envía frame a NPU --> C[🧠 Inferencia en tiempo real con RKNN];
        C -- Devuelve resultados --> B;
        B -- Guarda capturas + datos --> D[📁 Almacenamiento Local<br>/var/patients/];
    end

    subgraph "Cliente/Usuario"
       E[📱/💻 Navegador Web]
    end

    B -- Streaming MJPEG con inferencias --> E;
    E -- Envía comandos (capturar, etc.) --> B;

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#cde4f7,stroke:#333,stroke-width:2px    