# Sistema de Reconocimiento de Emociones con IA

Una aplicación de escritorio moderna y responsiva desarrollada en Python y PyQt6 para detectar emociones faciales en tiempo real a través de la cámara web, imágenes estáticas o archivos de video.

Este proyecto fue diseñado como una herramienta de apoyo para el entorno educativo de **SENATI**, permitiendo a los instructores obtener una mejor comprensión del estado emocional de los estudiantes.

---

## 📋 Características Principales

*   **Interfaz Gráfica Moderna:** UI intuitiva, responsiva y de aspecto profesional construida con PyQt6.
*   **Múltiples Fuentes de Análisis:**
    *   Cámara web en tiempo real.
    *   Archivos de imagen (JPG, PNG).
    *   Archivos de video (MP4, AVI).
*   **Selección de Modelos de Detección Facial:**
    *   **Haar Cascade:** Rápido y ligero, ideal para hardware con recursos limitados.
    *   **YOLOv5:** Alta precisión, recomendado para un seguimiento robusto.
    *   **MediaPipe:** Excelente balance entre rendimiento y precisión.
*   **Análisis de Emociones con DeepFace:** Utiliza la potente librería `DeepFace` para un análisis de emociones preciso, reconociendo 7 estados emocionales (feliz, triste, enojado, sorprendido, neutral, miedo, disgusto).
*   **Controles de Video Avanzados:** Controles de reproducción de video inspirados en YouTube, con barra de progreso, pausa, reanudación y control de tiempo.

---

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje:** Python 3.8
*   **Interfaz Gráfica:** PyQt6
*   **Procesamiento de Imagen/Video:** OpenCV
*   **Detección de Rostros:** Haar Cascade (OpenCV), YOLOv5 (PyTorch), MediaPipe
*   **Análisis de Emociones:** DeepFace

---

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Prerrequisitos

*   [Python 3.8](https://www.python.org/downloads/release/python-380/)
*   [Git](https://git-scm.com/downloads)
*   Una cámara web (para el análisis en tiempo real).

### 2. Instalación

**a. Clona el repositorio:**

```bash
git clone https://github.com/Yonsn76/emotion_recog.git
cd emotion_recog
```

**b. Crea y activa un entorno virtual (recomendado):**

```bash
# Crear el entorno
py -m venv venv

# Activar en Windows
.\venv\Scripts\activate
```

**c. Instala las dependencias:**

Asegúrate de que tu entorno virtual esté activado y luego ejecuta:

```bash
pip install -r requirements.txt
```

**d. Descarga el modelo YOLOv5 (Opcional pero recomendado para alta precisión):**

Si deseas utilizar el detector de rostros YOLOv5, descarga el archivo de pesos `yolov5x.pt` y colócalo en la carpeta raíz del proyecto.

*   Puedes descargarlo desde la [página de releases de YOLOv5](https://github.com/ultralytics/yolov5/releases).

---

## ▶️ Uso

Una vez que la instalación esté completa, puedes iniciar la aplicación.

1.  **Ejecuta el dashboard:**

    ```bash
    py dashboard.py
    ```

2.  **Interactúa con la interfaz:**
    *   **Selecciona un Modelo:** En la parte superior, elige el modelo de detección de rostros que prefieras (MediaPipe es el recomendado por defecto).
    *   **Activar Cámara:** Inicia la detección de emociones en tiempo real.
    *   **Subir Imagen:** Analiza una imagen estática.
    *   **Subir Video:** Carga un archivo de video para su análisis.

