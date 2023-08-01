import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Cargar el modelo
model = load_model('model.h5')

# Cargar el diccionario de clases
class_dict = np.load('class_indices.npy', allow_pickle=True).item()

# Invertir el diccionario de class_indices
class_dict = {v: k for k, v in class_dict.items()}

# Solicitar la ruta de la imagen al usuario
img_path = input('Por favor, introduce la ruta de la imagen a clasificar: ')

# Declarar las dimensiones de las imágenes
img_height, img_width = 100, 100

# Cargar la imagen
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Hacer la predicción
predictions = model.predict(img_batch)

# Imprimir la clase predicha
predicted_class = np.argmax(predictions[0])
print("Clase predicha:", class_dict[predicted_class])
