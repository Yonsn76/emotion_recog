import cv2
import numpy as np
import torch
from pathlib import Path
from deepface import DeepFace

class EmotionDetector:
    """
    Clase para detectar rostros y analizar emociones en imágenes y video.
    Utiliza diferentes modelos para la detección de rostros (Haar Cascade, YOLOv5, MediaPipe)
    y DeepFace para el análisis de emociones.
    """
    def __init__(self, model_type="haar"):
        """
        Inicializa el detector de emociones.
        Args:
            model_type (str): El modelo de detección de rostros a usar ('haar', 'yolo', 'mediapipe').
        """
        self.model_type = model_type
        self.emotion_translation = {
            'angry': 'Enojado',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Sorprendido',
            'neutral': 'Neutral'
        }
        self.mediapipe_available = False
        
        self.init_detectors()

    def init_detectors(self):
        """Inicializa los detectores de rostros según el modelo seleccionado."""
        if self.model_type == "haar":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif self.model_type == "yolo":
            try:
                from ultralytics import YOLO
                # Buscar el modelo en el mismo directorio que este archivo
                current_dir = Path(__file__).parent
                model_path = str(current_dir / 'yolov8n-face.pt')
                print(f"[EmotionDetector] Buscando modelo YOLOv8n-face en: {model_path}")
                
                # Cargar el modelo, si no existe se intentará descargar
                self.detector = YOLO(model_path)
            except Exception:
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
                print("MediaPipe no está instalado. Se utilizará Haar Cascade como alternativa.")
                self.model_type = "haar"
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        elif self.model_type == "deepface":
            # DeepFace se usa solo para el análisis de emoción independientemente del detector de rostros.
            pass

    def get_emotion(self, face_img):
        """
        Detecta la emoción en una imagen de rostro usando DeepFace.
        DeepFace se usa para el análisis de emoción independientemente del detector de rostros.
        """
        try:
            # Analiza la emoción en el rostro ya recortado.
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                emotion_en = result[0]['dominant_emotion']
            else:
                emotion_en = result['dominant_emotion']
            
            emotion = self.emotion_translation.get(emotion_en, "Neutral")
            return emotion, 1.0
        except Exception:
            # Si DeepFace falla (ej. rostro no claro), devuelve 'Neutral'.
            return "Neutral", 0.5

    def detect_faces_haar(self, frame):
        """Detecta rostros usando Haar Cascade y devuelve las coordenadas."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def detect_faces_yolo(self, frame):
        """Detecta rostros usando YOLOv8n-face (ultralytics) y devuelve las coordenadas de los rostros."""
        results = self.detector(frame)
        faces = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        faces.append((x1, y1, w, h))
        return faces

    def detect_faces_mediapipe(self, frame):
        """Detecta rostros usando MediaPipe y devuelve las coordenadas."""
        if not self.mediapipe_available:
            return []

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # El objeto FaceDetection NO es invocable, debe usarse .process()
        results = self.detector.process(img_rgb)

        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x, y, width, height = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                faces.append((x, y, width, height))
        return faces

    def process_frame(self, frame):
        """
        Procesa un frame de video o una imagen: detecta rostros y sus emociones.
        Si el modelo es YOLO, dibuja los resultados y muestra la emoción dominante sobre cada rostro.
        Optimizado para resolución media y fluidez.
        """
        if frame is None or frame.size == 0:
            return frame
        # --- Reescalar a resolución media (máx 640x480) ---
        h, w = frame.shape[:2]
        max_w, max_h = 640, 480
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        try:
            if self.model_type == "yolo":
                results = self.detector(frame)
                annotated = frame.copy()
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        # --- Reescalar rostro a 224x224 para DeepFace ---
                        face_resized = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                        try:
                            emotion = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)
                            emotion_label = emotion[0]['dominant_emotion'] if isinstance(emotion, list) else emotion['dominant_emotion']
                            emotion_label = self.emotion_translation.get(emotion_label.lower(), emotion_label)
                            color = (0, 140, 255)  # Naranja (BGR)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        except Exception:
                            cv2.putText(annotated, 'Error', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                return annotated
            elif self.model_type == "haar":
                faces = self.detect_faces_haar(frame)
                color = (255, 200, 100)  # Celeste (BGR)
            elif self.model_type == "mediapipe":
                faces = self.detect_faces_mediapipe(frame)
                color = (80, 220, 80)  # Verde (BGR)
            else:
                faces = []
                color = (0, 255, 0)
            for (x, y, w, h) in faces:
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                # --- Reescalar rostro a 224x224 para DeepFace ---
                face_resized = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                emotion, confidence = self.get_emotion(face_resized)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"{emotion}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception:
            # Solo mostrar error amigable, no traceback
            pass
        return frame

    def change_model(self, model_name):
        """
        Cambia dinámicamente el modelo de detección de rostros.
        Retorna True si el cambio fue exitoso, False en caso contrario.
        """
        if model_name not in ["haar", "yolo", "mediapipe"]:
            return False
        
        self.model_type = model_name
        try:
            self.init_detectors()
            return True
        except Exception as e:
            print(f"Error al cambiar al modelo {model_name}: {e}")
            # Si falla, revierte a Haar Cascade como modelo seguro.
            self.model_type = "haar"
            self.init_detectors()
            return False
