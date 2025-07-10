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
                # Carga el modelo YOLOv5 desde un archivo local o de torch.hub
                model_path = 'yolov5x.pt'
                if not Path(model_path).exists():
                    print(f"Modelo YOLO no encontrado en {model_path}, descargando...")
                    self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
                else:
                    self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')
                
                self.detector.classes = [0]  # Configura para detectar solo personas
                # Realiza una inferencia inicial para calentar el modelo y evitar cuelgues.
                _ = self.detector(torch.zeros(1, 3, 640, 640))
            except Exception as e:
                print(f"Error al cargar YOLOv5: {e}")
                print("Se utilizará Haar Cascade como alternativa.")
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
            # print(f"Advertencia en DeepFace: {e}")
            return "Neutral", 0.5

    def detect_faces_haar(self, frame):
        """Detecta rostros usando Haar Cascade y devuelve las coordenadas."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def detect_faces_yolo(self, frame):
        """Detecta rostros usando YOLOv5 y devuelve las coordenadas."""
        results = self.detector(frame)
        detections = results.xyxy[0].cpu().numpy()
        faces = []
        for *box, conf, cls in detections:
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
        Dibuja los resultados en el frame y lo devuelve.
        """
        if frame is None or frame.size == 0:
            return frame
            
        try:
            if self.model_type == "haar":
                faces = self.detect_faces_haar(frame)
            elif self.model_type == "yolo":
                faces = self.detect_faces_yolo(frame)
            elif self.model_type == "mediapipe":
                faces = self.detect_faces_mediapipe(frame)
            else:
                faces = []

            for (x, y, w, h) in faces:
                # Asegura que las coordenadas no excedan los límites del frame.
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                
                # Recorta el rostro para el análisis de emoción.
                face_img = frame[y:y+h, x:x+w]

                if face_img.size == 0:
                    continue

                emotion, confidence = self.get_emotion(face_img)
                
                # Dibuja el rectángulo y el texto de la emoción en el frame.
                color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
                
                text = f"{emotion}"
                
                # Calcula el tamaño del texto para el fondo.
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                
                # Dibuja un fondo oscuro para el texto.
                cv2.rectangle(frame, (x, y - text_height - 15), (x + text_width, y), (0, 0, 0), -1)
                
                # Dibuja el texto de la emoción.
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        except Exception as e:
            print(f"Error procesando frame: {e}")
            
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
