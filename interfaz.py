import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# D:\IA\class_indices.npy  D:\IA\class_indices.npy  D:\model.h5
modelo_path = 'model.h5'
classes_path = 'class_indices.npy'

# Verificar si el archivo del modelo existe
if not os.path.exists(modelo_path):
    print("Error: No se encontró el archivo del modelo. Asegúrate de que el archivo exista.")
    exit()

# Cargar el modelo
modelo = load_model(modelo_path)

# Intentar cargar las clases de etiquetas desde el archivo class_indices.npy
try:
    with open(classes_path, 'rb') as f:
        classes = np.load(f, allow_pickle=True).item()  # Convertir el diccionario a objeto de tipo dict
        print(classes)
except FileNotFoundError:
    print("Error: No se encontró el archivo class_indices.npy. Asegúrate de que el archivo exista y tenga los datos de las clases de etiquetas.")
    exit()

# Obtener la lista de dispositivos de video disponibles
video_devices = [i for i in range(10)]

# Mostrar las cámaras disponibles al usuario
print("Cámaras disponibles:")
for device_num in video_devices:
    cap = cv2.VideoCapture(device_num)
    if cap.isOpened():
        print(f"Cámara {device_num}")

# Pedir al usuario que ingrese el número de la cámara que desea usar
while True:
    try:
        camera_num = int(input("Ingresa el número de la cámara que deseas usar: "))
        if camera_num in video_devices:
            break
        else:
            print("Número de cámara no válido. Por favor, intenta nuevamente.")
    except ValueError:
        print("Por favor, ingresa un número válido.")

# Iniciar la cámara seleccionada
cap = cv2.VideoCapture(camera_num)

# Definir las dimensiones de las imágenes
alto_img, ancho_img = 100, 100

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionar la imagen
    frame_resized = cv2.resize(frame, (alto_img, ancho_img))

    # Normalizar la imagen dividiéndola por 255
    img_normalized = frame_resized / 255.0

    # Convertir la imagen a una matriz y expandir las dimensiones para que coincida con el formato de entrada del modelo
    img_array = img_to_array(img_normalized)
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Hacer la predicción
    predictions = modelo.predict(img_array_expanded)

    # Obtener la etiqueta de clase predicha
    indice_clase_predicha = np.argmax(predictions, axis=1)[0]
    etiqueta_clase_predicha = list(classes.keys())[indice_clase_predicha]

    # Mostrar la etiqueta de la clase predicha en el video
    cv2.putText(frame, etiqueta_clase_predicha, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar el frame resultante
    cv2.imshow('Video', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo esté hecho, liberar la captura
cap.release()
cv2.destroyAllWindows()
