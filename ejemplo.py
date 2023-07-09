import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Declarar las dimensiones de las imágenes y el tamaño del batch
img_height, img_width = 150, 150
batch_size = 32

# Cargar los datos de entrenamiento
train_data_gen = ImageDataGenerator(rescale=1./255)
train_data = train_data_gen.flow_from_directory(
    'data/entrenamiento',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Cargar los datos de validación
val_data_gen = ImageDataGenerator(rescale=1./255)
val_data = val_data_gen.flow_from_directory(
    'data/validacion',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Crear el modelo
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 10
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size
)

# Solicitar la ruta de la imagen al usuario
img_path = input('Por favor, introduce la ruta de la imagen a clasificar: ')

# Cargar la imagen
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Hacer la predicción
predictions = model.predict(img_batch)

# Diccionario de clases: invierte el diccionario de class_indices
class_dict = {v: k for k, v in train_data.class_indices.items()}

# Imprimir la clase predicha
predicted_class = np.argmax(predictions[0])
print("Clase predicha:", class_dict[predicted_class])
