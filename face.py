import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from deepface import DeepFace

def analizar_emocion_frame(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]

        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emoción: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def analizar_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se puede acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = analizar_emocion_frame(frame)
        cv2.imshow("Emociones desde webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analizar_video():
    path = filedialog.askopenfilename(title="Seleccionar archivo de video", filetypes=[("Archivos de video", "*.mp4 *.avi *.mov")])
    if not path:
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se puede abrir el archivo de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = analizar_emocion_frame(frame)
        cv2.imshow("Emociones desde video", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analizar_imagen():
    path = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
    if not path:
        return

    frame = cv2.imread(path)
    if frame is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        return

    frame = analizar_emocion_frame(frame)
    cv2.imshow("Emociones desde imagen", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Análisis de Emociones con DeepFace")
root.geometry("300x200")

# Crear botones
btn_webcam = tk.Button(root, text="Analizar desde Webcam", command=analizar_webcam, width=30)
btn_video = tk.Button(root, text="Analizar desde Video", command=analizar_video, width=30)
btn_imagen = tk.Button(root, text="Analizar desde Imagen", command=analizar_imagen, width=30)

# Posicionar los botones
btn_webcam.pack(pady=10)
btn_video.pack(pady=10)
btn_imagen.pack(pady=10)

# Iniciar el bucle de la interfaz
root.mainloop()
