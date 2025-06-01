import cv2
import numpy as np
import torch
from pathlib import Path
from deepface import DeepFace

class EmotionDetector:
    def __init__(self, model_type="haar"):
        self.model_type = model_type
        self.emotions = ['Feliz', 'Triste', 'Enojado', 'Sorprendido', 'Neutral']
        self.mediapipe_available = False
        
        # Inicializar detectores
        self.init_detectors()

    def init_detectors(self):
        """Inicializa los detectores según el modelo seleccionado"""
        if self.model_type == "haar":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif self.model_type == "yolo":
            try:
                # Cargar YOLOv5 desde el archivo de pesos
                model_path = 'yolov5x.pt'
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
                self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')
                self.detector.classes = [0]  # Solo detectar personas
                # Intentar warmup para evitar errores de controlador
                _ = self.detector(torch.zeros(1, 3, 640, 640))
            except Exception as e:
                print(f"Error al cargar YOLO: {e}")
                print("Usando Haar Cascade como alternativa.")
                self.model_type = "haar"
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        elif self.model_type == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                self.mediapipe_available = True
            except ImportError:
                print("MediaPipe no está disponible. Usando Haar Cascade como alternativa.")
                self.model_type = "haar"
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        elif self.model_type == "deepface":
            pass

    def get_emotion(self, face_img):
        """Detecta la emoción en una imagen de rostro usando el modelo seleccionado"""
        if self.model_type == "deepface":
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    emotion = result[0]['dominant_emotion']
                else:
                    emotion = result['dominant_emotion']
                return emotion, 1.0
            except Exception as e:
                print(f"Error en DeepFace: {e}")
                return "Neutral", 0.5
        else:
            # Para otros modelos, intentamos una detección simple basada en expresiones faciales
            try:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                # Usar un clasificador simple para emociones (puede ser entrenado o predefinido)
                # Aquí se usa un placeholder para detección simple
                # Por ejemplo, usar un clasificador LBPH o similar si está disponible
                # Por ahora, se usa una heurística simple basada en la intensidad de píxeles
                mean_intensity = np.mean(gray)
                if mean_intensity > 130:
                    emotion = 'Feliz'
                elif mean_intensity > 100:
                    emotion = 'Neutral'
                else:
                    emotion = 'Triste'
                confidence = 0.7
                return emotion, confidence
            except Exception as e:
                print(f"Error en detección simple: {e}")
                emotion = np.random.choice(self.emotions)
                confidence = 0.5
                return emotion, confidence

    def detect_faces_haar(self, frame):
        """Detecta rostros usando Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Mejoramos la detección ajustando los parámetros
        if len(faces) == 0:
            # Si no se detectan rostros, intentamos con parámetros más permisivos
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

    def detect_faces_yolo(self, frame):
        """Detecta rostros usando YOLOv5"""
        results = self.detector(frame)
        faces = []
        for det in results.xyxy[0]:
            if det[-1] == 0:  # Si es una persona
                x1, y1, x2, y2 = map(int, det[:4])
                faces.append((x1, y1, x2, y2))
        return faces

    def detect_faces_mediapipe(self, frame):
        """Detecta rostros usando MediaPipe"""
        if not self.mediapipe_available:
            return self.detect_faces_haar(frame)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                faces.append((x1, y1, x2, y2))
        return faces

    def detect_faces_deepface(self, frame):
        """Detecta rostros y emociones usando DeepFace"""
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            faces = []
            if isinstance(analysis, list):
                for face in analysis:
                    region = face['region']
                    x1, y1 = region['x'], region['y']
                    x2, y2 = x1 + region['w'], y1 + region['h']
                    faces.append((x1, y1, x2, y2))
            else:
                region = analysis['region']
                x1, y1 = region['x'], region['y']
                x2, y2 = x1 + region['w'], y1 + region['h']
                faces.append((x1, y1, x2, y2))
            return faces, analysis
        except Exception as e:
            print(f"Error en DeepFace: {e}")
            return [], None

    def process_frame(self, frame):
        """Procesa un frame para detectar rostros y emociones"""
        if frame is None or frame.size == 0:
            return frame

        # Hacer una copia del frame para no modificar el original
        output_frame = frame.copy()
        
        if self.model_type == "haar":
            faces = self.detect_faces_haar(frame)
            analysis = None
        elif self.model_type == "yolo":
            faces = self.detect_faces_yolo(frame)
            analysis = None
        elif self.model_type == "deepface":
            faces, analysis = self.detect_faces_deepface(frame)
        else:  # mediapipe
            faces = self.detect_faces_mediapipe(frame)
            analysis = None

        for idx, (x1, y1, x2, y2) in enumerate(faces):
            # Extraer región del rostro con un margen
            margin = 20
            y1_m = max(0, y1 - margin)
            y2_m = min(frame.shape[0], y2 + margin)
            x1_m = max(0, x1 - margin)
            x2_m = min(frame.shape[1], x2 + margin)
            face = frame[y1_m:y2_m, x1_m:x2_m]
            
            if face.size == 0:
                continue

            # Obtener emoción
            if self.model_type == "deepface" and analysis:
                if isinstance(analysis, list) and len(analysis) > idx:
                    emotion = analysis[idx]['dominant_emotion']
                elif not isinstance(analysis, list):
                    emotion = analysis['dominant_emotion']
                else:
                    emotion, conf = self.get_emotion(face)
                conf = 1.0
            else:
                emotion, conf = self.get_emotion(face)

            # Dibujar bbox y etiqueta con estilo mejorado
            if self.model_type == "deepface":
                color = (0, 0, 255)  # Rojo para DeepFace
                text_color = (255, 255, 255)  # Blanco para el texto
                thickness = max(2, int((x2-x1) / 50))  # Texto más grueso para DeepFace
            else:
                color = (0, 255, 0)  # Verde para otros modelos
                text_color = (255, 255, 255)  # Blanco para el texto
                thickness = max(1, int((x2-x1) / 100))
            
            # Dibujar bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Preparar el texto
            label = f'{emotion} ({conf:.2f})'
            font_scale = (x2-x1) / 200  # Escala proporcional al tamaño del rostro
            if self.model_type == "deepface":
                font_scale = min(max(0.7, font_scale), 1.2)  # Límites más altos para DeepFace
            else:
                font_scale = min(max(0.5, font_scale), 1.0)  # Límites normales para otros modelos
            
            # Obtener el tamaño del texto para el fondo
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Dibujar fondo para el texto
            cv2.rectangle(output_frame, 
                        (x1, y1-text_height-baseline-5),
                        (x1+text_width, y1),
                        color, -1)
            
            # Dibujar el texto
            cv2.putText(output_frame, label,
                       (x1, y1-baseline-5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, text_color, thickness)

        return output_frame

    def change_model(self, model_type):
        """Cambia el modelo de detección"""
        if model_type in ["haar", "yolo", "mediapipe", "deepface"]:
            self.model_type = model_type
            self.init_detectors()
            return True
        return False

    def is_mediapipe_available(self):
        """Verifica si MediaPipe está disponible"""
        return self.mediapipe_available
