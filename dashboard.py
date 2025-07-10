import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QFrame, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QSlider, QLayout, QScrollArea
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QIcon
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, QRect, QPoint, QSize

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
    """Un frame que contiene los controles de reproducción de video."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoControls")
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(10)

        # Botón de Play/Pausa
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.clicked.connect(self.update_play_pause_symbol)

        # Botón de retroceso
        self.rewind_btn = QPushButton("«")

        # Botón de avance
        self.forward_btn = QPushButton("»")

        # Botón de detener
        self.stop_btn = QPushButton("⏹")

        # Botón de pantalla completa
        self.fullscreen_btn = QPushButton("⛶")

        # Etiqueta de tiempo
        self.time_label = QLabel("00:00 / 00:00")

        # Deslizador de progreso
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)

        # Añadir widgets al layout
        layout.addWidget(self.rewind_btn)
        layout.addWidget(self.play_pause_btn)
        layout.addWidget(self.forward_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_slider, 1)
        layout.addWidget(self.fullscreen_btn)

        self.setLayout(layout)
        self.setFixedHeight(35)

        # Hoja de estilo moderna, similar a YouTube
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
        """Actualiza el ícono del botón entre play y pausa."""
        if self.play_pause_btn.isChecked():
            self.play_pause_btn.setText("⏸")
        else:
            self.play_pause_btn.setText("▶")

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
        
        self.init_ui()
        self.mediapipe_btn.setChecked(True)

    def showEvent(self, event):
        """Fuerza la actualización del layout en la primera visualización para corregir la posición inicial."""
        super().showEvent(event)
        # Difiere la actualización para permitir que el bucle de eventos procese los eventos iniciales.
        QTimer.singleShot(0, self.layout.update)

    def init_ui(self):
        # Estilo de la ventana principal
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
                padding: 5px;
                margin: 0px;
                border-radius: 8px;
                border: 1px solid #333333;
            }
            QLabel#PreviewImage {
                background-color: #1a1a1a;
                border-radius: 8px;
                border: 2px solid #444444;
                padding: 2px;
                margin: 0px;
                min-height: 600px;
            }
        """)


        # --- Scroll Area para el layout principal ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        # Widget contenedor para el contenido principal
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(25, 15, 25, 15)

        # --- Contenedor para todos los controles colapsables ---
        self.collapsible_controls_frame = QFrame()
        collapsible_controls_layout = QVBoxLayout(self.collapsible_controls_frame)
        collapsible_controls_layout.setContentsMargins(0, 0, 0, 0)
        collapsible_controls_layout.setSpacing(15)

        # --- Sección de Selección del Detector de Rostros ---
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
        

        self.layout.addWidget(self.preview_frame, 1)

        # Configura el scroll area
        self.scroll_area.setWidget(self.main_widget)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)
        self.setMinimumSize(1200, 800)  # Tamaño mínimo más grande

        # --- Animación para los controles superiores ---
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
        if self.detector.change_model(model_name):
            self.current_model = model_name
            # Actualiza el estado visual de los botones de modelo.
            self.haar_btn.setChecked(model_name == "haar")
            self.yolo_btn.setChecked(model_name == "yolo")
            self.mediapipe_btn.setChecked(model_name == "mediapipe")
        else:
            QMessageBox.warning(self, "Error", f"Modelo desconocido: {model_name}")

    def toggle_webcam(self):
        """Inicia o detiene la captura de la cámara web."""
        if self.is_webcam_active:
            self._stop_current_media()
        else:
            self._stop_current_media()

            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "Error", "No se pudo acceder a la cámara")
                    return
                    
                # Configurar resolución de la cámara
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.is_webcam_active = True
                self.webcam_btn.setText("Detener Cámara")
                self.timer.start(30)
                self.video_controls.setVisible(False)
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

        try:
            frame = self.detector.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = self.scale_image_to_label(qt_image)
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return
        
        # Actualiza el deslizador de progreso y la etiqueta de tiempo para videos.
        if self.cap and self.total_frames > 0 and not self.is_webcam_active:
            try:
                self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                # Actualiza el deslizador.
                progress = int((self.current_frame / self.total_frames) * 100) if self.total_frames > 0 else 0
                self.video_controls.progress_slider.setValue(progress)

                # Actualiza la etiqueta de tiempo.
                if self.video_fps > 0:
                    current_sec = self.current_frame / self.video_fps
                    total_sec = self.total_frames / self.video_fps
                    
                    current_time_str = f"{int(current_sec // 60):02}:{int(current_sec % 60):02}"
                    total_time_str = f"{int(total_sec // 60):02}:{int(total_sec % 60):02}"
                    
                    self.video_controls.time_label.setText(f"{current_time_str} / {total_time_str}")
            except Exception as e:
                print(f"Error actualizando controles de video: {e}")

    def upload_image(self):
        """Abre un diálogo para que el usuario seleccione una imagen y la procesa."""
        self._stop_current_media()

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
        """Abre un diálogo para que el usuario seleccione un video y comienza la reproducción."""
        self._stop_current_media()
        
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
                    
                # Obtiene las propiedades del video.
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                # Validación de propiedades del video
                if self.total_frames <= 0:
                    QMessageBox.warning(self, "Error", "El archivo de video parece estar corrupto")
                    self.cap.release()
                    self.cap = None
                    return
                    
                if self.video_fps <= 0:
                    self.video_fps = 30  # FPS por defecto
                
                self.current_frame = 0
                self.video_controls.progress_slider.setValue(0)
                
                self.webcam_btn.setText("Activar Cámara")
                self.video_controls.setVisible(True)
                self.video_paused = False
                self.video_controls.play_pause_btn.setChecked(True)
                self.video_controls.update_play_pause_symbol()
                self.timer.start(int(1000 / self.video_fps))
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
        """Se ejecuta cuando la ventana cambia de tamaño para redimensionar la imagen."""
        super().resizeEvent(event)
        # Si hay una imagen cargada y tenemos el pixmap original, la reescala
        if (hasattr(self, '_original_pixmap') and 
            self._original_pixmap and not self._original_pixmap.isNull()):
            scaled_pixmap = self._original_pixmap.scaled(
                self.image_label.size() - QSize(10, 10),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDashboard()
    window.showMaximized()
    sys.exit(app.exec())
