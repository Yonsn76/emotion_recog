import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QFrame, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QSlider
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QIcon
from PyQt6.QtCore import QTimer, Qt
from emotion_detector import EmotionDetector

class ModelButton(QPushButton):
    def __init__(self, text, description, parent=None):
        super().__init__(parent)
        self.setObjectName("ModelButton")
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(text)
        title.setStyleSheet("color: #FFFFFF; font-weight: 600; font-size: 16px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = QLabel(description)
        desc.setStyleSheet("color: #CCCCCC; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)

        container = QWidget()
        container.setLayout(layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(160)
        self.setMinimumHeight(50)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #FFFFFF;
                border: 2px solid #444444;
                border-radius: 15px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: 600;
                height: 50px;
                margin: 0px;
                min-width: 120px;
                text-align: center;
            }
            QPushButton:checked {
                background-color: #e84118;
                color: #FFFFFF;
                border: 2px solid #e84118;
                padding: 8px 20px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #ff5c33;
                border: 2px solid #ff5c33;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(232, 65, 24))
        shadow.setOffset(0)
        self.setGraphicsEffect(shadow)

class InputButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("InputButton")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(140)
        self.setMinimumHeight(45)
        self.setStyleSheet("""
            QPushButton {
                background-color: #e84118;
                color: #FFFFFF;
                border: none;
                border-radius: 22px;
                padding: 10px 24px;
                font-size: 16px;
                font-weight: 700;
                height: 45px;
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
                box-shadow: 0 0 15px #e84118;
            }
            QPushButton[primary="true"]:hover {
                background-color: #ff5c33;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(232, 65, 24))
        shadow.setOffset(0)
        self.setGraphicsEffect(shadow)

class VideoControls(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoControls")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Progress bar (slider)
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #2a2a2a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e84118;
                border: 1px solid #e84118;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #e84118;
                border-radius: 4px;
            }
        """)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Play/Pause button using Unicode symbols
        self.play_pause_btn = QPushButton("⏵")  # Play symbol
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.clicked.connect(self.update_play_pause_symbol)
        
        # Stop button
        self.stop_btn = QPushButton("⏹")
        
        # Rewind button
        self.rewind_btn = QPushButton("⏪")
        
        # Forward button
        self.forward_btn = QPushButton("⏩")
        
        # Style for all control buttons
        control_button_style = """
            QPushButton {
                background-color: #2a2a2a;
                border: 2px solid #444444;
                border-radius: 20px;
                padding: 5px;
                min-width: 40px;
                min-height: 40px;
                font-size: 20px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #e84118;
            }
            QPushButton:pressed {
                background-color: #e84118;
            }
        """
        
        for btn in [self.play_pause_btn, self.stop_btn, self.rewind_btn, self.forward_btn]:
            btn.setStyleSheet(control_button_style)
            buttons_layout.addWidget(btn)
        
        buttons_layout.addStretch()
        
        layout.addWidget(self.progress_slider)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def update_play_pause_symbol(self):
        if self.play_pause_btn.isChecked():
            self.play_pause_btn.setText("⏸")  # Pause symbol
        else:
            self.play_pause_btn.setText("⏵")  # Play symbol

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
        
        self.init_ui()

    def init_ui(self):
        # Set window style
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #FFFFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QFrame#HeaderFrame {
                background-color: #282c34;
                border-bottom: 1px solid #444444;
                min-height: 60px;
                max-height: 60px;
                padding: 15px 40px;
            }
            QLabel#HeaderTitle {
                color: #FFFFFF;
                font-size: 24px;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            QLabel#SectionTitle {
                color: #FFFFFF;
                font-size: 20px;
                font-weight: 700;
                padding: 20px 16px 12px;
            }
            QFrame#PreviewSection {
                background-color: #2a2a3d;
                padding: 20px;
                margin: 0px;
                border-radius: 15px;
                border: 1px solid #444444;
            }
            QLabel#PreviewImage {
                background-color: #2a2a3d;
                border-radius: 15px;
                border: none;
                padding: 12px;
                margin: 0px;
            }
        """)

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(40, 20, 40, 20)

        # Model Selection Section
        model_section = QFrame()
        model_section.setObjectName("ModelSection")
        model_layout = QVBoxLayout(model_section)
        model_layout.setSpacing(12)
        model_title = QLabel("Elige un Modelo")
        model_title.setObjectName("SectionTitle")
        model_layout.addWidget(model_title)
        model_buttons_layout = QHBoxLayout()
        model_buttons_layout.setSpacing(12)
        self.haar_btn = ModelButton("Haar Cascade", "Ligero y eficiente")
        self.yolo_btn = ModelButton("YOLOv5", "Rápido y preciso")
        self.deepface_btn = ModelButton("DeepFace", "Basado en aprendizaje profundo")
        self.mediapipe_btn = ModelButton("MediaPipe", "Detección facial de MediaPipe")
        self.haar_btn.setChecked(True)
        self.haar_btn.setStyleSheet(self.haar_btn.styleSheet() + "background-color: #3498db; border-color: #2980b9;")
        self.yolo_btn.setStyleSheet(self.yolo_btn.styleSheet() + "background-color: #e67e22; border-color: #d35400;")
        self.deepface_btn.setStyleSheet(self.deepface_btn.styleSheet() + "background-color: #9b59b6; border-color: #8e44ad;")
        self.mediapipe_btn.setStyleSheet(self.mediapipe_btn.styleSheet() + "background-color: #2ecc71; border-color: #27ae60;")
        self.haar_btn.clicked.connect(lambda: self.change_model("haar"))
        self.yolo_btn.clicked.connect(lambda: self.change_model("yolo"))
        self.deepface_btn.clicked.connect(lambda: self.change_model("deepface"))
        self.mediapipe_btn.clicked.connect(lambda: self.change_model("mediapipe"))
        model_buttons_layout.addWidget(self.haar_btn)
        model_buttons_layout.addWidget(self.yolo_btn)
        model_buttons_layout.addWidget(self.deepface_btn)
        model_buttons_layout.addWidget(self.mediapipe_btn)
        model_layout.addLayout(model_buttons_layout)
        self.layout.addWidget(model_section)

        # Input Selection Section
        input_section = QFrame()
        input_section.setObjectName("InputSection")
        input_layout = QVBoxLayout(input_section)
        input_layout.setSpacing(12)
        input_title = QLabel("Selecciona Entrada")
        input_title.setObjectName("SectionTitle")
        input_layout.addWidget(input_title)
        input_buttons_layout = QHBoxLayout()
        input_buttons_layout.setSpacing(12)
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
        input_buttons_layout.addStretch()
        input_layout.addLayout(input_buttons_layout)
        self.layout.addWidget(input_section)

        # Preview Area
        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("PreviewSection")
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel()
        self.image_label.setObjectName("PreviewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        preview_layout.addWidget(self.image_label)
        
        # Add video controls
        self.video_controls = VideoControls()
        self.video_controls.setVisible(False)  # Hide initially
        preview_layout.addWidget(self.video_controls)
        
        # Connect video control signals
        self.video_controls.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.video_controls.stop_btn.clicked.connect(self.stop_video)
        self.video_controls.rewind_btn.clicked.connect(self.rewind_video)
        self.video_controls.forward_btn.clicked.connect(self.forward_video)
        self.video_controls.progress_slider.sliderMoved.connect(self.seek_video)
        
        self.layout.addWidget(self.preview_frame, 1)
        self.setLayout(self.layout)
        self.setMinimumSize(900, 700)

    def change_model(self, model_name):
        if self.detector.change_model(model_name):
            self.current_model = model_name
            # Actualizar el estado de los botones
            self.haar_btn.setChecked(model_name == "haar")
            self.yolo_btn.setChecked(model_name == "yolo")
            self.deepface_btn.setChecked(model_name == "deepface")
            self.mediapipe_btn.setChecked(model_name == "mediapipe")

            # Efecto visual de selección
            self.haar_btn.setStyleSheet(self.haar_btn.styleSheet().split("QPushButton:checked")[0] + """
                QPushButton:checked {
                    background-color: #3498db;
                    color: #FFFFFF;
                    border: 2px solid #2980b9;
                    font-weight: 800;
                }
            """)
            self.yolo_btn.setStyleSheet(self.yolo_btn.styleSheet().split("QPushButton:checked")[0] + """
                QPushButton:checked {
                    background-color: #e67e22;
                    color: #FFFFFF;
                    border: 2px solid #d35400;
                    font-weight: 800;
                }
            """)
            self.deepface_btn.setStyleSheet(self.deepface_btn.styleSheet().split("QPushButton:checked")[0] + """
                QPushButton:checked {
                    background-color: #9b59b6;
                    color: #FFFFFF;
                    border: 2px solid #8e44ad;
                    font-weight: 800;
                }
            """)
            self.mediapipe_btn.setStyleSheet(self.mediapipe_btn.styleSheet().split("QPushButton:checked")[0] + """
                QPushButton:checked {
                    background-color: #2ecc71;
                    color: #FFFFFF;
                    border: 2px solid #27ae60;
                    font-weight: 800;
                }
            """)

            if self.is_webcam_active:
                self.toggle_webcam()
        else:
            QMessageBox.warning(self, "Error", f"Modelo desconocido: {model_name}")

    def toggle_webcam(self):
        if self.is_webcam_active:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.is_webcam_active = False
            self.webcam_btn.setText("Activar Cámara")
            self.image_label.clear()
            self.video_controls.setVisible(False)
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "No se pudo acceder a la cámara")
                return
            self.is_webcam_active = True
            self.webcam_btn.setText("Detener Cámara")
            self.timer.start(30)
            self.video_controls.setVisible(True)

    def update_frame(self):
        if self.video_paused:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_webcam()
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
        
        # Update progress slider for video playback
        if self.cap and not self.is_webcam_active:
            self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((self.current_frame / self.total_frames) * 100)
            self.video_controls.progress_slider.setValue(progress)

    def upload_image(self):
        if self.is_webcam_active:
            self.toggle_webcam()
        
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
        if self.is_webcam_active:
            self.toggle_webcam()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Video",
            "",
            "Archivos de Video (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "No se pudo cargar el video")
                return
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            
            self.is_webcam_active = True
            self.webcam_btn.setText("Detener")
            self.video_controls.setVisible(True)
            self.video_paused = False
            self.video_controls.play_pause_btn.setChecked(False)
            self.timer.start(30)

    def toggle_play_pause(self):
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.timer.stop()
        else:
            self.timer.start(30)

    def stop_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            self.video_paused = True
            self.video_controls.play_pause_btn.setChecked(True)
            self.timer.stop()

    def rewind_video(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current - 30)  # Rewind 30 frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.current_frame = new_pos
            progress = int((new_pos / self.total_frames) * 100)
            self.video_controls.progress_slider.setValue(progress)

    def forward_video(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = min(self.total_frames - 1, current + 30)  # Forward 30 frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.current_frame = new_pos
            progress = int((new_pos / self.total_frames) * 100)
            self.video_controls.progress_slider.setValue(progress)

    def seek_video(self, value):
        if self.cap and not self.is_webcam_active:
            frame_pos = int((value / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame = frame_pos

    def closeEvent(self, event):
        if self.is_webcam_active and self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDashboard()
    window.show()
    sys.exit(app.exec())
