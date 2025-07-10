import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QFrame, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QSlider, QLayout, QScrollArea, QMenu
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QIcon
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QRect, QPoint, QSize, QEvent
from pathlib import Path
from emotion_detector import EmotionDetector

class FlowLayout(QLayout):
    """Un layout personalizado que organiza widgets en un flujo, similar al texto."""
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
        """Calcula la altura necesaria para un ancho determinado."""
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
        """La lógica principal que posiciona los widgets en el layout."""
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
    """Botón personalizado para seleccionar los modelos de detección."""
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
    """Botón de estilo personalizado para las opciones de entrada."""
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
        layout.setSpacing(2)

        self.rewind_btn = QPushButton("«")
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.clicked.connect(self.update_play_pause_symbol)
        self.forward_btn = QPushButton("»")
        self.stop_btn = QPushButton("⏹")
        self.settings_btn = QPushButton("⚙️")
        self.fullscreen_btn = QPushButton("⛶")
        self.time_label = QLabel("00:00 / 00:00")
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)

        layout.addWidget(self.rewind_btn)
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.forward_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.settings_btn)
        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_slider, 1)
        layout.addWidget(self.fullscreen_btn)

        self.setLayout(layout)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QFrame#VideoControls {
                background-color: #181818;
                border: none;
                border-radius: 10px;
            }
            QPushButton {
                background-color: transparent;
                color: #fff;
                border: none;
                font-size: 22px;
                padding: 0 2px;
                min-width: 28px;
                min-height: 28px;
                border-radius: 6px;
                transition: background 0.2s;
            }
            QPushButton:hover {
                background-color: #232323;
                color: #e84118;
            }
            QLabel {
                color: #fff;
                font-size: 15px;
                font-weight: 600;
                min-width: 90px;
                qproperty-alignment: AlignCenter;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: #444;
                margin: 0 0 0 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e84118;
                border: none;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #e84118;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #222;
                border-radius: 3px;
            }
        """)

        from PyQt6.QtWidgets import QMenu
        self.speed_menu = QMenu()
        self.speed_menu.addAction("0.5x", lambda: self.parent().set_playback_speed(0.5))
        self.speed_menu.addAction("1x", lambda: self.parent().set_playback_speed(1.0))
        self.speed_menu.addAction("1.5x", lambda: self.parent().set_playback_speed(1.5))
        self.speed_menu.addAction("2x", lambda: self.parent().set_playback_speed(2.0))
        self.settings_btn.setMenu(self.speed_menu)

    def update_play_pause_symbol(self):
        if self.play_pause_btn.isChecked():
            self.play_pause_btn.setText("⏸")
        else:
            self.play_pause_btn.setText("▶")

class CameraSidebar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(110)
        self.setStyleSheet("""
            QFrame {
                background: #232323;
                border-top-right-radius: 20px;
                border-bottom-right-radius: 20px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 40, 0, 40)
        layout.setSpacing(32)
        self.record_btn = QPushButton()
        self.record_btn.setFixedSize(48, 48)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: #fff;
                border: 4px solid #fff;
                border-radius: 24px;
                color: #e84118;
            }
            QPushButton:pressed {
                background: #e84118;
                color: #fff;
            }
        """)
        self.record_btn.setIcon(QIcon("icons/rec1.ico"))
        self.record_btn.setIconSize(QSize(32, 32))
        self.capture_btn = QPushButton()
        self.capture_btn.setFixedSize(72, 72)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background: #fff;
                border: 4px solid #fff;
                border-radius: 36px;
                color: #181818;
            }
            QPushButton:pressed {
                background: #3498db;
                color: #fff;
            }
        """)
        self.capture_btn.setIcon(QIcon("icons/camara.ico"))
        self.capture_btn.setIconSize(QSize(48, 48))
        self.fullscreen_btn = QPushButton()
        self.fullscreen_btn.setFixedSize(48, 48)
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background: #fff;
                border: 4px solid #fff;
                border-radius: 24px;
                color: #232323;
                font-size: 26px;
            }
            QPushButton:pressed {
                background: #e84118;
                color: #fff;
            }
        """)
        self.fullscreen_btn.setText("⛶")
        layout.addStretch(1)
        layout.addWidget(self.record_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.capture_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.fullscreen_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(2)
        self.setLayout(layout)

class EmotionDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Emociones")
        self.detector = EmotionDetector(model_type="mediapipe")
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_webcam_active = False
        self.current_model = "mediapipe"
        self.video_paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 0
        self.is_fullscreen = False
        self.controls_visible_before_fullscreen = True
        self.is_recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.init_ui()
        self.mediapipe_btn.setChecked(True)

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #FFFFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
        """)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(25, 15, 25, 15)
        self.collapsible_controls_frame = QFrame()
        collapsible_controls_layout = QVBoxLayout(self.collapsible_controls_frame)
        collapsible_controls_layout.setContentsMargins(0, 0, 0, 0)
        collapsible_controls_layout.setSpacing(15)

        # --- Contenedor para todos los controles colapsables ---
        face_detector_section = QFrame()
        face_detector_section.setObjectName("ModelSection")
        face_detector_layout = QVBoxLayout(face_detector_section)
        face_detector_layout.setSpacing(10)
        face_detector_title = QLabel("Elige un Detector de Rostros")
        face_detector_title.setObjectName("SectionTitle")
        face_detector_layout.addWidget(face_detector_title)
        
        face_buttons_layout = FlowLayout()
        face_buttons_layout.setSpacing(10)
        
        self.haar_btn = ModelButton("Haar Cascade", "Ligero y rápido")
        self.haar_btn.setProperty("model", "haar")
        self.yolo_btn = ModelButton("YOLOv8n-face", "Optimizado para rostros, rápido y preciso")
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

        # --- Sección del Detector de Emociones (Solo visualización) ---
        emotion_detector_section = QFrame()
        emotion_detector_layout = QVBoxLayout(emotion_detector_section)
        emotion_detector_layout.setSpacing(10)
        emotion_title = QLabel("Detector de Emociones")
        emotion_title.setObjectName("SectionTitle")
        
        emotion_model_label = QLabel("DeepFace")
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

        # --- Sección de Selección de Entrada ---
        input_section = QFrame()
        input_section.setObjectName("InputSection")
        input_layout = QVBoxLayout(input_section)
        input_layout.setSpacing(10)
        input_title = QLabel("Selecciona Entrada")
        input_title.setObjectName("SectionTitle")
        input_layout.addWidget(input_title)
        input_buttons_layout = FlowLayout()
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
        input_layout.addLayout(input_buttons_layout)
        collapsible_controls_layout.addWidget(input_section)

        # Añade el contenedor de controles al layout principal.
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

        # --- Área de Previsualización ---
        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("PreviewSection")
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setSpacing(5)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        self.image_label = QLabel()
        self.image_label.setObjectName("PreviewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setScaledContents(False)  # Mantener aspecto
        preview_layout.addWidget(self.image_label, 1)  # Stretch factor 1 para expandir
        
        # Añade los controles de video con el nuevo diseño.
        self.video_controls = VideoControls()
        self.video_controls.setVisible(False)
        preview_layout.addWidget(self.video_controls, 0)  # No stretch para controles
        
        # Conecta las señales de los controles de video.
        self.video_controls.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.video_controls.rewind_btn.clicked.connect(self.rewind_video)
        self.video_controls.forward_btn.clicked.connect(self.forward_video)
        self.video_controls.stop_btn.clicked.connect(self.stop_video)
        self.video_controls.progress_slider.sliderMoved.connect(self.seek_video)
        self.video_controls.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        
        self.record_btn = QPushButton("⏺️ Grabar")
        self.capture_btn = QPushButton("📸 Foto")
        self.fullscreen_cam_btn = QPushButton("⛶")
        self.record_btn.setFixedHeight(40)
        self.capture_btn.setFixedHeight(40)
        self.fullscreen_cam_btn.setFixedHeight(40)
        self.record_btn.setStyleSheet("background-color: #e84118; color: #fff; font-weight: 700; border-radius: 8px;")
        self.capture_btn.setStyleSheet("background-color: #3498db; color: #fff; font-weight: 700; border-radius: 8px;")
        self.fullscreen_cam_btn.setStyleSheet("background-color: #232323; color: #fff; font-weight: 700; border-radius: 8px;")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.capture_btn.clicked.connect(self.capture_image)
        self.fullscreen_cam_btn.clicked.connect(self.toggle_fullscreen)
        self.layout.insertWidget(2, QWidget())
        self.layout.itemAt(2).widget().setLayout(QHBoxLayout()) 

        self.layout.addWidget(self.preview_frame, 1)

        # Configura el scroll area
        self.scroll_area.setWidget(self.main_widget)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area, 1)
        self.setLayout(main_layout)
        self.setMinimumSize(1200, 800)  # Tamaño mínimo más grande

        # --- Animación para los controles superiores ---
        self.animation = QPropertyAnimation(self.collapsible_controls_frame, b"maximumHeight")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self.camera_sidebar = CameraSidebar(self.image_label)
        self.camera_sidebar.setMinimumHeight(220)
        self.camera_sidebar.setFixedWidth(110)
        self.camera_sidebar.record_btn.setFixedSize(48, 48)
        self.camera_sidebar.capture_btn.setFixedSize(72, 72)
        self.camera_sidebar.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.camera_sidebar.record_btn.clicked.connect(self.toggle_recording)
        self.camera_sidebar.capture_btn.clicked.connect(self.capture_image)
        self.camera_sidebar.hide()
        self.record_time_label = QLabel(self.image_label)
        self.record_time_label.setStyleSheet("background: rgba(0,0,0,0.7); color: #fff; font-size: 20px; font-weight: bold; border-radius: 8px; padding: 6px 16px;")
        self.record_time_label.move(20, 20)
        self.record_time_label.hide()
        self.record_seconds = 0
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.update_record_time)

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
        
        # Oculta el frame cuando la animación de colapso termina.
        if end_height == 0:
            self.animation.finished.connect(lambda: self.collapsible_controls_frame.setVisible(False))
        else:
            # Desconecta la señal para evitar que se oculte al expandir.
            try:
                self.animation.finished.disconnect()
            except (TypeError, RuntimeError):
                pass

        self.animation.start()

    def toggle_fullscreen(self):
        """Activa o desactiva el modo de pantalla completa."""
        if self.is_fullscreen:
            self.showNormal()
            # Restaura la visibilidad de los controles.
            self.toggle_button.setVisible(True)
            if self.controls_visible_before_fullscreen:
                self.collapsible_controls_frame.setVisible(True)
            
            self.is_fullscreen = False
            self.video_controls.fullscreen_btn.setText("⛶")
        else:
            self.is_fullscreen = True
            # Guarda el estado de visibilidad de los controles.
            self.controls_visible_before_fullscreen = self.collapsible_controls_frame.isVisible()
            
            self.showFullScreen()
            # Oculta otros widgets para una experiencia inmersiva.
            self.toggle_button.setVisible(False)
            self.collapsible_controls_frame.setVisible(False)
            self.video_controls.fullscreen_btn.setText("Exit")

    def keyPressEvent(self, event):
        """Maneja el evento de presionar una tecla, para salir de pantalla completa con ESC."""
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
        """Cambia el modelo de detección de rostros."""
        if model_name == "yolo":
            try:
                from ultralytics import YOLO
                current_dir = Path(__file__).parent
                model_path = str(current_dir / 'yolov8n-face.pt')
                print(f"[Dashboard] Buscando modelo YOLOv8n-face en: {model_path}")
                if not Path(model_path).exists():
                    print("[Dashboard] Modelo yolov8n-face.pt no encontrado, se intentará descargar...")
                self.detector.detector = YOLO(model_path)
                self.detector.model_type = "yolo"
                self.current_model = "yolo"
                self.haar_btn.setChecked(False)
                self.yolo_btn.setChecked(True)
                self.mediapipe_btn.setChecked(False)
                print(f"YOLOv8n-face modelo carga exitoso {model_path}")
            except Exception as e:
                error_msg = f"No se pudo cargar yolov8n-face.pt: {str(e)}"
                print(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
                self.change_model("haar")
        else:
            if self.detector.change_model(model_name):
                self.current_model = model_name
                self.haar_btn.setChecked(model_name == "haar")
                self.yolo_btn.setChecked(model_name == "yolo")
                self.mediapipe_btn.setChecked(model_name == "mediapipe")
            else:
                QMessageBox.warning(self, "Error", f"Modelo desconocido: {model_name}")

    def toggle_webcam(self):
        if self.is_webcam_active:
            self._stop_current_media()
            self.camera_sidebar.hide()
        else:
            self._stop_current_media()
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "Error", "No se pudo acceder a la cámara")
                    return
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_webcam_active = True
                self.webcam_btn.setText("Detener Cámara")
                self.timer.start(30)
                self.video_controls.setVisible(False)
                self.camera_sidebar.show()
                self.position_camera_sidebar()
                self.camera_sidebar.raise_()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error accediendo a la cámara: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def update_frame(self):
        """Se ejecuta con cada tick del QTimer para procesar y mostrar un nuevo frame."""
        if self.video_paused and not self.is_webcam_active:
            return
        if not self.cap or not self.cap.isOpened():
            self.timer.stop()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            if self.is_webcam_active:
                self.toggle_webcam()
            else: # Fin del archivo de video
                self.stop_video()
            return
        # Asegura que el frame no esté vacío.
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        max_w, max_h = 640, 480
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if self.is_webcam_active and self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        try:
            frame = self.detector.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = self.scale_image_to_label(qt_image)
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error procesando frame: {str(e)}")
            return
        if self.cap and self.total_frames > 0 and not self.is_webcam_active:
            try:
                self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress = int((self.current_frame / self.total_frames) * 100) if self.total_frames > 0 else 0
                self.video_controls.progress_slider.setValue(progress)
                if self.video_fps > 0:
                    current_sec = self.current_frame / self.video_fps
                    total_sec = self.total_frames / self.video_fps
                    current_time_str = f"{int(current_sec // 60):02}:{int(current_sec % 60):02}"
                    total_time_str = f"{int(total_sec // 60):02}:{int(total_sec % 60):02}"
                    self.video_controls.time_label.setText(f"{current_time_str} / {total_time_str}")
            except Exception as e:
                pass

    def upload_image(self):
        self._stop_current_media()
        self.camera_sidebar.hide()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    QMessageBox.warning(self, "Error", "No se pudo cargar la imagen")
                    return
                h, w = image.shape[:2]
                max_w, max_h = 640, 480
                scale = min(max_w / w, max_h / h, 1.0)
                if scale < 1.0:
                    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                processed = self.detector.process_frame(image)
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                h, w, ch = processed.shape
                bytes_per_line = ch * w
                qt_image = QImage(processed.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_pixmap = self.scale_image_to_label(qt_image)
                self.image_label.setPixmap(scaled_pixmap)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error procesando la imagen: {str(e)}")

    def upload_video(self):
        self._stop_current_media()
        self.camera_sidebar.hide()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Video",
            "",
            "Archivos de Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            try:
                self.is_webcam_active = False
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "Error", "No se pudo cargar el video")
                    return
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.total_frames <= 0:
                    QMessageBox.warning(self, "Error", "El archivo de video parece estar corrupto")
                    self.cap.release()
                    self.cap = None
                    return
                if self.video_fps <= 0:
                    self.video_fps = 30
                self.current_frame = 0
                self.video_controls.progress_slider.setValue(0)
                self.webcam_btn.setText("Activar Cámara")
                self.video_controls.setVisible(True)
                self.video_paused = False
                self.video_controls.play_pause_btn.setChecked(True)
                self.video_controls.update_play_pause_symbol()
                self.timer.start(int(1000 / self.video_fps))
                self.camera_sidebar.hide()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando el video: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def toggle_play_pause(self):
        """Pausa o reanuda la reproducción del video."""
        if not self.is_webcam_active and self.cap:
            if self.video_paused:
                self.timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 30)
                self.video_paused = False
            else:
                self.timer.stop()
                self.video_paused = True
            self.video_controls.update_play_pause_symbol()

    def stop_video(self):
        """Detiene el video y lo reinicia al principio."""
        self.timer.stop()
        if self.cap and self.total_frames > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            self.video_paused = True
            self.video_controls.play_pause_btn.setChecked(False)
            self.video_controls.update_play_pause_symbol()
            # Muestra el primer frame.
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                frame = self.detector.process_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_pixmap = self.scale_image_to_label(qt_image)
                self.image_label.setPixmap(scaled_pixmap)
            # Resetea la posición después de mostrar el primer frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    def rewind_video(self):
        """Retrocede el video 5 segundos."""
        if self.cap and self.total_frames > 0:
            fps = self.video_fps if self.video_fps > 0 else 30
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current - (5 * fps))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.update_frame()

    def forward_video(self):
        """Adelanta el video 5 segundos."""
        if self.cap and self.total_frames > 0:
            fps = self.video_fps if self.video_fps > 0 else 30
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = min(self.total_frames - 1, current + (5 * fps))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.update_frame()

    def seek_video(self, value):
        """Busca una posición específica en el video según el valor del deslizador."""
        if self.cap and self.total_frames > 0:
            frame_pos = int((value / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame = frame_pos
            self.update_frame()

    def closeEvent(self, event):
        """Se asegura de liberar los recursos al cerrar la aplicación."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_camera_sidebar()
        if (hasattr(self, '_original_pixmap') and 
            self._original_pixmap and not self._original_pixmap.isNull()):
            scaled_pixmap = self._original_pixmap.scaled(
                self.image_label.size() - QSize(10, 10),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        if self.is_webcam_active:
            self.camera_sidebar.show()
            self.position_camera_sidebar()
            self.camera_sidebar.raise_()
        else:
            self.camera_sidebar.hide()

    def scale_image_to_label(self, qt_image):
        """Escala la imagen para que ocupe el máximo espacio disponible manteniendo la proporción."""
        pixmap = QPixmap.fromImage(qt_image)
        
        # Guarda el pixmap original para redimensionamiento posterior
        self._original_pixmap = pixmap
        
        # Obtiene el tamaño disponible del label (con un pequeño margen)
        available_size = self.image_label.size()
        margin = 10
        target_size = QSize(available_size.width() - margin, available_size.height() - margin)
        
        # Escala la imagen manteniendo la proporción
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        return scaled_pixmap

    def toggle_recording(self):
        if not self.is_webcam_active:
            return
        if not self.is_recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_path, _ = QFileDialog.getSaveFileName(self, "Guardar Video", "video.avi", "Archivos de Video (*.avi)")
            if not save_path:
                return
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            self.is_recording = True
            self.camera_sidebar.record_btn.setIcon(QIcon("icons/rec2.ico"))
            self.camera_sidebar.record_btn.setIconSize(QSize(32, 32))
            self.record_seconds = 0
            self.update_record_time()
            self.record_time_label.show()
            self.record_timer.start(1000)
        else:
            self.is_recording = False
            self.camera_sidebar.record_btn.setIcon(QIcon("icons/rec1.ico"))
            self.camera_sidebar.record_btn.setIconSize(QSize(32, 32))
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_time_label.hide()
            self.record_timer.stop()

    def update_record_time(self):
        self.record_seconds += 1
        mins = self.record_seconds // 60
        secs = self.record_seconds % 60
        self.record_time_label.setText(f"REC {mins:02}:{secs:02}")

    def capture_image(self):
        if not self.is_webcam_active:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None and frame.size > 0:
            save_path, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen", "captura.jpg", "Archivos de Imagen (*.jpg *.png *.jpeg)")
            if save_path:
                cv2.imwrite(save_path, frame)

    def position_floating_panel(self):
        w = self.width()
        h = self.height()
        panel_w = self.sidebar.width()
        panel_h = self.sidebar.height()
        x = w - panel_w - 20
        y = (h - panel_h) // 2
        self.sidebar.move(x, y)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseMove:
            if self.is_webcam_active:
                pass # No need to show floating panel here
                # self.floating_panel.show_panel()
                # self.floating_panel.hide_timer.start()
        return super().eventFilter(obj, event)

    def position_camera_sidebar(self):
        il_w = self.image_label.width()
        il_h = self.image_label.height()
        sb_w = self.camera_sidebar.width()
        sb_h = self.camera_sidebar.height()
        x = il_w - sb_w - 10
        y = (il_h - sb_h) // 2
        self.camera_sidebar.move(x, y)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDashboard()
    window.showMaximized()
    sys.exit(app.exec())
