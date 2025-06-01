# Sistema de Reconocimiento de Emociones

Este proyecto implementa un sistema de reconocimiento de emociones con una interfaz gráfica moderna que permite detectar emociones tanto en tiempo real (webcam) como en imágenes y videos, utilizando múltiples modelos de detección facial.

## Características

- Interfaz gráfica moderna con PyQt6
- Múltiples modelos de detección facial:
  * **Haar Cascade**: Rápido y ligero, ideal para sistemas con recursos limitados
  * **YOLOv5**: Alta precisión, modelo rápido y preciso
  * **DeepFace**: Basado en aprendizaje profundo
  * **MediaPipe**: Buen balance entre precisión y rendimiento
- Clasificación de 5 emociones básicas:
  - Felicidad
  - Tristeza
  - Enojo
  - Sorpresa
  - Neutral
- Soporte para webcam en tiempo real
- Soporte para análisis de imágenes y videos
- Efecto visual de selección para cada modelo
- Mensajes y textos en español

## Requisitos

- Python 3.8 o superior
- OpenCV
- NumPy
- PyQt6
- PyTorch y torchvision (para YOLO/DeepFace)
- Otras dependencias listadas en requirements.txt
- Webcam (para detección en tiempo real)


2. Descarga el modelo yolov5x.pt y colócalo en la carpeta principal del proyecto (si usas YOLO).

3. Instala las dependencias:
```bash
py -m pip install -r requirements.txt
```

## Uso

1. Ejecuta el dashboard:
```bash
py dashboard.py
```

2. En la interfaz gráfica, tienes las siguientes opciones:
   - Selecciona el modelo de detección (Haar, YOLOv5, DeepFace, MediaPipe)
   - Haz clic en "Activar Cámara" para iniciar/detener la detección en tiempo real
   - Haz clic en "Subir Imagen" para seleccionar una imagen y detectar emociones en ella
   - Haz clic en "Subir Video" para analizar emociones en un video

3. Para cerrar el programa:
   - Cierra la ventana del dashboard
   - O presiona Ctrl+C en la terminal
