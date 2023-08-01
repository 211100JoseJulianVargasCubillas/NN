import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Verificar las dispositivos disponibles
devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print("GPU encontrada.")
    for device in devices:
        print("Dispositivo:", device)
else:
    print("No se encontró ninguna GPU.")

# Verificar si TensorFlow está utilizando la GPU por defecto
print("TensorFlow está utilizando la GPU por defecto:", tf.test.is_built_with_cuda())
print("TensorFlow está utilizando la GPU:", tf.config.list_physical_devices('GPU'))

# Declarar las dimensiones de las imágenes y el tamaño del batch
img_height, img_width = 100, 100
batch_size = 32

# Obtener la lista de todas las imágenes de entrenamiento
train_data_dir = 'data/entrenamiento'
train_image_files = []
for root, _, files in os.walk(train_data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_image_files.append(os.path.join(root, file))

# Dividir aleatoriamente las imágenes en conjuntos de entrenamiento y validación
random.shuffle(train_image_files)
num_validation_samples = int(0.4 * len(train_image_files))
train_image_files = train_image_files[num_validation_samples:]
validation_image_files = train_image_files[:num_validation_samples]

# Crear un dataframe con las rutas de las imágenes y sus etiquetas
train_df = pd.DataFrame(train_image_files, columns=['filename'])
train_df['label'] = train_df['filename'].apply(lambda x: os.path.basename(os.path.dirname(x)))

validation_df = pd.DataFrame(validation_image_files, columns=['filename'])
validation_df['label'] = validation_df['filename'].apply(lambda x: os.path.basename(os.path.dirname(x)))

# Obtener el número de clases del generador de imágenes
num_classes = len(np.unique(train_df['label']))

# Cargar los datos de entrenamiento
train_data_gen = ImageDataGenerator(rescale=1./255)
train_data = train_data_gen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Cargar los datos de validación
val_data_gen = ImageDataGenerator(rescale=1./255)
val_data = val_data_gen.flow_from_dataframe(
    validation_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
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
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')  # Usar num_classes en lugar de train_data.num_classes
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 50
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size
)

# Guardar el modelo
model.save('model.h5')

# Guardar el diccionario de clases
np.save('class_indices.npy', train_data.class_indices)

# Obtener las etiquetas verdaderas del conjunto de validación
true_labels = val_data.classes

# Obtener las predicciones del modelo en el conjunto de validación
predictions = model.predict(val_data)
predicted_labels = np.argmax(predictions, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Mostrar la matriz de confusión
classes = list(train_data.class_indices.keys())
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap='viridis', values_format='d')

# Mostrar el plot de la matriz de confusión
import matplotlib.pyplot as plt
plt.title('Matriz de Confusión')
plt.show()
