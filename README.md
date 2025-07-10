# Detector de Emociones

**Recomendado usar Python 3.8**

## Descripción

Esta es una aplicación de escritorio moderna y responsiva desarrollada en Python y PyQt6 para detectar emociones faciales en tiempo real a través de la cámara web, imágenes estáticas o archivos de video.

Este proyecto fue diseñado como una herramienta de apoyo para el entorno educativo, permitiendo a los instructores obtener una mejor comprensión del estado emocional de los estudiantes.

## Características principales
- Detección de emociones en imágenes, videos y cámara web.
- Permite grabar video y tomar fotos directamente desde la cámara.
- Interfaz gráfica moderna y fácil de usar (PyQt6).
- Soporte para modelos de detección de rostro: Haar Cascade, YOLOv8n-face, MediaPipe.

## Requisitos
- Python 3.8 (recomendado)
- Ver dependencias en `requirements.txt`

## Uso
1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta la aplicación:
   ```bash
   python dashboard.py
   ```

## Funcionalidades de cámara
- **Grabar video:** Puedes grabar video desde la cámara web y guardarlo en tu equipo.
- **Tomar foto:** Puedes capturar una imagen desde la cámara web y guardarla como archivo.

## Modelos soportados
- Haar Cascade
- YOLOv8n-face (requiere el archivo `yolov8n-face.pt` en el directorio del proyecto)
- MediaPipe

## Créditos
- DeepFace
- Ultralytics YOLO
- MediaPipe
- PyQt6

---

## 📋 Características Principales

*   **Interfaz Gráfica Moderna:** UI intuitiva, responsiva y de aspecto profesional construida con PyQt6.
*   **Múltiples Fuentes de Análisis:**
    *   Cámara web en tiempo real.
    *   Archivos de imagen (JPG, PNG).
    *   Archivos de video (MP4, AVI).
*   **Selección de Modelos de Detección Facial:**
    *   **Haar Cascade:** Rápido y ligero, ideal para hardware con recursos limitados.
    *   **YOLOv8n-face:** Optimizado para detección de rostros, recomendado para alta precisión y velocidad.
    *   **MediaPipe:** Excelente balance entre rendimiento y precisión.
*   **Análisis de Emociones con DeepFace:** Utiliza la librería `DeepFace` para un análisis de emociones preciso, reconociendo 7 estados emocionales (feliz, triste, enojado, sorprendido, neutral, miedo, disgusto).
*   **Controles de Video Avanzados:** Controles de reproducción de video inspirados en YouTube, con barra de progreso, pausa, reanudación y control de tiempo.

---

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje:** Python 3.8
*   **Interfaz Gráfica:** PyQt6
*   **Procesamiento de Imagen/Video:** OpenCV
*   **Detección de Rostros:** Haar Cascade (OpenCV), YOLOv8n-face (PyTorch/ONNX), MediaPipe
*   **Análisis de Emociones:** DeepFace

---

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Prerrequisitos

*   [Python 3.8](https://www.python.org/downloads/release/python-380/)
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


**d. Descarga el modelo YOLOv8n-face (recomendado para detección de rostros):**

Repositorio oficial:
```bash
https://github.com/derronqi/yolov8-face
```
Enlace de descarga directa del modelo: 
```bash
https://drive.usercontent.google.com/u/0/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb&export=download
```
Coloca el archivo `yolov8n-face.pt` en la carpeta raíz del proyecto.

Este modelo está optimizado específicamente para la detección de rostros y es recomendado para obtener mejores resultados.

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

