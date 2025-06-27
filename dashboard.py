import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QFrame, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QSlider, QLayout
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QIcon
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QRect, QPoint, QSize

from emotion_detector import EmotionDetector

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        margin, _, _, _ = self.getContentsMargins()
        size += QSize(2 * margin, 2 * margin)
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0

        for item in self.itemList:
            style = self.parentWidget().style() if self.parentWidget() else QApplication.style()
            space_x = self.spacing() + style.layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Horizontal)
            space_y = self.spacing() + style.layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Vertical)
            
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

class ModelButton(QPushButton):
    def __init__(self, text, description, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelButton")
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel(text)
        title.setStyleSheet("color: #FFFFFF; font-weight: 600; font-size: 15px; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = QLabel(description)
        desc.setStyleSheet("color: #CCCCCC; font-size: 11px; background: transparent;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()
        
        self.setLayout(layout)

        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(130)
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            QPushButton#ModelButton {
                background-color: #2a2a2a;
                color: #FFFFFF;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 5px;
                font-size: 15px;
                font-weight: 600;
                text-align: center;
            }
            QPushButton#ModelButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #555555;
            }
            QPushButton#ModelButton[model="haar"]:checked {
                background-color: #3498db;
                border-color: #2980b9;
                font-weight: 700;
            }
            QPushButton#ModelButton[model="yolo"]:checked {
                background-color: #e67e22;
                border-color: #d35400;
                font-weight: 700;
            }
            QPushButton#ModelButton[model="mediapipe"]:checked {
                background-color: #2ecc71;
                border-color: #27ae60;
                font-weight: 700;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

class InputButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("InputButton")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(120)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QPushButton {
                background-color: #e84118;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 15px;
                font-weight: 700;
                height: 40px;
                margin: 0px;
                min-width: 120px;
                letter-spacing: 0.02em;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #ff5c33;
            }
            QPushButton[primary="true"] {
                background-color: #e84118;
                color: #FFFFFF;
                font-weight: 800;
            }
            QPushButton[primary="true"]:hover {
                background-color: #ff5c33;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(232, 65, 24, 150))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

class VideoControls(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoControls")
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(10)

        # Play/Pause button
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.clicked.connect(self.update_play_pause_symbol)

        # Rewind button
        self.rewind_btn = QPushButton("«")

        # Forward button
        self.forward_btn = QPushButton("»")

        # Stop button
        self.stop_btn = QPushButton("⏹")

        # Fullscreen button
        self.fullscreen_btn = QPushButton("⛶") # Unicode for fullscreen

        # Time label
        self.time_label = QLabel("00:00 / 00:00")

        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)

        # Add widgets to layout
        layout.addWidget(self.rewind_btn)
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.forward_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_slider, 1)  # Stretch factor of 1
        layout.addWidget(self.fullscreen_btn)

        self.setLayout(layout)
        self.setFixedHeight(35)

        # Modern, YouTube-like stylesheet
        self.setStyleSheet("""
            QFrame#VideoControls {
                background-color: transparent;
                border: none;
            }
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                font-size: 18px;
                padding: 0px;
                min-width: 25px;
            }
            QPushButton:hover {
                color: #e84118;
            }
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: 500;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 5px;
                background: #4a4a4a;
                margin: 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #e84118;
                border: none;
                width: 14px;
                height: 14px;
                margin: -4.5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #e84118;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: #6a6a6a;
                border-radius: 2px;
            }
        """)

    def update_play_pause_symbol(self):
        if self.play_pause_btn.isChecked():
            self.play_pause_btn.setText("⏸")  # Pause symbol
        else:
            self.play_pause_btn.setText("▶")  # Play symbol

class EmotionDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Emociones")
        self.detector = EmotionDetector(model_type="haar")
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_webcam_active = False
        self.current_model = "haar"
        self.video_paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 0
        self.is_fullscreen = False
        self.controls_visible_before_fullscreen = True
        
        self.init_ui()
        self.haar_btn.setChecked(True)

    def showEvent(self, event):
        """Force layout update on first show to fix initial positioning."""
        super().showEvent(event)
        # Defer the layout update to allow the main event loop to process initial events
        QTimer.singleShot(0, self.layout.update)

    def init_ui(self):
        # Set window style
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #FFFFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QFrame#HeaderFrame {
                background-color: #282c34;
                border-bottom: 1px solid #333333;
                min-height: 50px;
                max-height: 50px;
                padding: 10px 25px;
            }
            QLabel#HeaderTitle {
                color: #FFFFFF;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#SectionTitle {
                color: #FFFFFF;
                font-size: 18px;
                font-weight: 600;
                padding: 15px 0 8px;
            }
            QFrame#PreviewSection {
                background-color: #2a2a2a;
                padding: 15px;
                margin: 0px;
                border-radius: 8px;
                border: 1px solid #333333;
            }
            QLabel#PreviewImage {
                background-color: #2a2a2a;
                border-radius: 8px;
                border: none;
                padding: 0px;
                margin: 0px;
            }
        """)

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(25, 15, 25, 15)

        # --- Contenedor para todos los controles colapsables ---
        self.collapsible_controls_frame = QFrame()
        collapsible_controls_layout = QVBoxLayout(self.collapsible_controls_frame)
        collapsible_controls_layout.setContentsMargins(0, 0, 0, 0)
        collapsible_controls_layout.setSpacing(15)

        # --- Face Detector Selection Section ---
        face_detector_section = QFrame()
        face_detector_section.setObjectName("ModelSection")
        face_detector_layout = QVBoxLayout(face_detector_section)
        face_detector_layout.setSpacing(10)
        face_detector_title = QLabel("Elige un Detector de Rostros")
        face_detector_title.setObjectName("SectionTitle")
        face_detector_layout.addWidget(face_detector_title)
        
        face_buttons_layout = FlowLayout() # <-- Cambio a FlowLayout
        face_buttons_layout.setSpacing(10)
        
        self.haar_btn = ModelButton("Haar Cascade", "Ligero y rápido")
        self.haar_btn.setProperty("model", "haar")
        self.yolo_btn = ModelButton("YOLOv5", "Rápido y preciso")
        self.yolo_btn.setProperty("model", "yolo")
        self.mediapipe_btn = ModelButton("MediaPipe", "Moderno y robusto")
        self.mediapipe_btn.setProperty("model", "mediapipe")

        self.haar_btn.clicked.connect(lambda: self.change_model("haar"))
        self.yolo_btn.clicked.connect(lambda: self.change_model("yolo"))
        self.mediapipe_btn.clicked.connect(lambda: self.change_model("mediapipe"))

        face_buttons_layout.addWidget(self.haar_btn)
        face_buttons_layout.addWidget(self.yolo_btn)
        face_buttons_layout.addWidget(self.mediapipe_btn)
        
        face_detector_layout.addLayout(face_buttons_layout)
        collapsible_controls_layout.addWidget(face_detector_section)

        # --- Emotion Detector Section (Display Only) ---
        emotion_detector_section = QFrame()
        emotion_detector_layout = QVBoxLayout(emotion_detector_section)
        emotion_detector_layout.setSpacing(10)
        emotion_title = QLabel("Detector de Emociones")
        emotion_title.setObjectName("SectionTitle")
        
        emotion_model_label = QLabel("DeepFace (Análisis de Emociones Real)")
        emotion_model_label.setStyleSheet("""
            color: #FFFFFF; 
            font-size: 15px; 
            font-weight: 600;
            background-color: #2a2a2a;
            border: 1px solid #9b59b6;
            border-radius: 8px;
            padding: 8px 15px;
            height: 45px;
        """)
        emotion_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        emotion_detector_layout.addWidget(emotion_title)
        emotion_detector_layout.addWidget(emotion_model_label)
        collapsible_controls_layout.addWidget(emotion_detector_section)

        # --- Input Selection Section ---
        input_section = QFrame()
        input_section.setObjectName("InputSection")
        input_layout = QVBoxLayout(input_section)
        input_layout.setSpacing(10)
        input_title = QLabel("Selecciona Entrada")
        input_title.setObjectName("SectionTitle")
        input_layout.addWidget(input_title)
        input_buttons_layout = FlowLayout() # <-- Cambio a FlowLayout
        input_buttons_layout.setSpacing(10)
        self.webcam_btn = InputButton("Activar Cámara")
        self.webcam_btn.setProperty("primary", True)
        self.upload_img_btn = InputButton("Subir Imagen")
        self.upload_video_btn = InputButton("Subir Video")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        self.upload_img_btn.clicked.connect(self.upload_image)
        self.upload_video_btn.clicked.connect(self.upload_video)
        input_buttons_layout.addWidget(self.webcam_btn)
        input_buttons_layout.addWidget(self.upload_video_btn)
        input_buttons_layout.addWidget(self.upload_img_btn)
        # input_buttons_layout.addStretch() # No es necesario con FlowLayout
        input_layout.addLayout(input_buttons_layout)
        collapsible_controls_layout.addWidget(input_section)

        # Se añade el contenedor principal de controles al layout de la ventana.
        self.layout.addWidget(self.collapsible_controls_frame)

        # --- Botón para Ocultar/Mostrar Controles ---
        self.toggle_button = QPushButton("Ocultar Controles ▲")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self.toggle_top_controls)
        self.layout.addWidget(self.toggle_button)

        # --- Input Selection Section (MOVIDA) ---
        # Esta sección ahora está dentro de 'self.collapsible_controls_frame'

        # Preview Area
        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("PreviewSection")
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setSpacing(10)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        self.image_label = QLabel()
        self.image_label.setObjectName("PreviewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        preview_layout.addWidget(self.image_label)
        
        # Add video controls (NUEVO DISEÑO)
        self.video_controls = VideoControls()
        self.video_controls.setVisible(False)  # Hide initially
        preview_layout.addWidget(self.video_controls)
        
        # Connect video control signals
        self.video_controls.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.video_controls.rewind_btn.clicked.connect(self.rewind_video)
        self.video_controls.forward_btn.clicked.connect(self.forward_video)
        self.video_controls.stop_btn.clicked.connect(self.stop_video)
        self.video_controls.progress_slider.sliderMoved.connect(self.seek_video)
        self.video_controls.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        
        self.layout.addWidget(self.preview_frame, 1)
        self.setLayout(self.layout)
        self.setMinimumSize(850, 650)

        # --- Animación para los controles superiores ---
        # La animación ahora se aplica al contenedor de todos los controles colapsables.
        self.animation = QPropertyAnimation(self.collapsible_controls_frame, b"maximumHeight")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuart)

    def toggle_top_controls(self):
        start_height = self.collapsible_controls_frame.height()
        end_height = 0
        
        if not self.collapsible_controls_frame.isVisible():
            self.collapsible_controls_frame.setVisible(True)
            end_height = self.collapsible_controls_frame.sizeHint().height()
            self.toggle_button.setText("Ocultar Controles ▲")
        else:
            self.toggle_button.setText("Mostrar Controles ▼")

        self.animation.setStartValue(start_height)
        self.animation.setEndValue(end_height)
        
        # Hide the frame after the animation finishes if we are collapsing it
        if end_height == 0:
            # Use a lambda to avoid issues with disconnecting a non-existent connection
            self.animation.finished.connect(lambda: self.collapsible_controls_frame.setVisible(False))
        else:
            # Disconnect any previous connection to avoid hiding on expand
            try:
                self.animation.finished.disconnect()
            except (TypeError, RuntimeError):
                pass # No connection to disconnect or object already deleted

        self.animation.start()

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
            # Restaurar la visibilidad de los controles
            self.toggle_button.setVisible(True)
            if self.controls_visible_before_fullscreen:
                self.collapsible_controls_frame.setVisible(True)
            
            self.is_fullscreen = False
            self.video_controls.fullscreen_btn.setText("⛶")
        else:
            self.is_fullscreen = True
            # Guardar el estado actual de los controles
            self.controls_visible_before_fullscreen = self.collapsible_controls_frame.isVisible()
            
            self.showFullScreen()
            # Ocultar otros widgets
            self.toggle_button.setVisible(False)
            self.collapsible_controls_frame.setVisible(False)
            self.video_controls.fullscreen_btn.setText("Exit")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def _stop_current_media(self):
        """Detiene cualquier fuente de medios activa (cámara web o video)."""
        if self.is_webcam_active:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_webcam_active = False
            self.webcam_btn.setText("Activar Cámara")
            self.image_label.clear()
            self.video_controls.setVisible(False)
            self.video_controls.time_label.setText("00:00 / 00:00")
        elif self.timer.isActive():
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_controls.setVisible(False)

    def change_model(self, model_name):
        if self.detector.change_model(model_name):
            self.current_model = model_name
            # Actualizar el estado de los botones
            self.haar_btn.setChecked(model_name == "haar")
            self.yolo_btn.setChecked(model_name == "yolo")
            self.mediapipe_btn.setChecked(model_name == "mediapipe")

            # No es necesario reiniciar la cámara web, el modelo se aplicará en el siguiente frame.
            # if self.is_webcam_active:
            #     self.toggle_webcam() # Esto detenía y reiniciaba, innecesario.
        else:
            QMessageBox.warning(self, "Error", f"Modelo desconocido: {model_name}")

    def toggle_webcam(self):
        if self.is_webcam_active:
            self._stop_current_media()
        else:
            self._stop_current_media()  # Detener cualquier video/imagen antes de iniciar la cámara

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "No se pudo acceder a la cámara")
                return
            self.is_webcam_active = True
            self.webcam_btn.setText("Detener Cámara")
            self.timer.start(30)
            self.video_controls.setVisible(False) # Hide video controls for webcam

    def update_frame(self):
        if self.video_paused and not self.is_webcam_active:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            if self.is_webcam_active:
                self.toggle_webcam()
            else: # End of video file
                self.stop_video()
            return
        
        # Asegurarse de que el frame no esté vacío
        if frame is None or frame.size == 0:
            return

        frame = self.detector.process_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        # Update progress slider and time label for video playback
        if self.cap and self.total_frames > 0 and not self.is_webcam_active:
            self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Update slider
            progress = int((self.current_frame / self.total_frames) * 100) if self.total_frames > 0 else 0
            self.video_controls.progress_slider.setValue(progress)

            # Update time label
            if self.video_fps > 0:
                current_sec = self.current_frame / self.video_fps
                total_sec = self.total_frames / self.video_fps
                
                current_time_str = f"{int(current_sec // 60):02}:{int(current_sec % 60):02}"
                total_time_str = f"{int(total_sec // 60):02}:{int(total_sec % 60):02}"
                
                self.video_controls.time_label.setText(f"{current_time_str} / {total_time_str}")

    def upload_image(self):
        self._stop_current_media()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Error", "No se pudo cargar la imagen")
                return
            
            processed = self.detector.process_frame(image)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = processed.shape
            bytes_per_line = ch * w
            qt_image = QImage(processed.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def upload_video(self):
        self._stop_current_media()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Video",
            "",
            "Archivos de Video (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.is_webcam_active = False # Ensure webcam mode is off
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "No se pudo cargar el video")
                return
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            
            self.webcam_btn.setText("Activar Cámara")
            self.video_controls.setVisible(True)
            self.video_paused = False
            self.video_controls.play_pause_btn.setChecked(True) # Set to playing state
            self.video_controls.update_play_pause_symbol()
            self.timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 30)

    def toggle_play_pause(self):
        if not self.is_webcam_active and self.cap:
            if self.video_paused:
                self.timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 30)
                self.video_paused = False
            else:
                self.timer.stop()
                self.video_paused = True
            self.video_controls.update_play_pause_symbol()

    def stop_video(self):
        self.timer.stop()
        if self.cap and self.total_frames > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            self.video_paused = True
            self.video_controls.play_pause_btn.setChecked(False) # Set to paused state
            self.video_controls.update_play_pause_symbol()
            # Show first frame
            self.update_frame()
            self.video_paused = True


    def rewind_video(self):
        if self.cap and self.total_frames > 0:
            fps = self.video_fps if self.video_fps > 0 else 30
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current - (5 * fps))  # Rewind 5 seconds
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.update_frame()

    def forward_video(self):
        if self.cap and self.total_frames > 0:
            fps = self.video_fps if self.video_fps > 0 else 30
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = min(self.total_frames - 1, current + (5 * fps))  # Forward 5 seconds
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.update_frame()

    def seek_video(self, value):
        if self.cap and self.total_frames > 0:
            frame_pos = int((value / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame = frame_pos
            self.update_frame()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDashboard()
    window.showMaximized() # Show the window maximized
    sys.exit(app.exec())
