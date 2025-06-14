import os
import tensorflow as tf
import tensorflow.keras as keras
import pathlib
from tensorflow.keras import layers
from PIL import Image 
import numpy as np


#Função com o dataset
if __name__ == "__main__":
  
    data_dir = pathlib.Path(r"C:\Users\Vinic\Downloads\archive\train")
    batch_size = 32
    img_height = 48
    img_width = 48
#pré-processamento com data_augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
#dataset que usa as variaveis de antes
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
#diretório de teste
    test_ds = tf.keras.utils.image_dataset_from_directory(
        pathlib.Path(r"C:\Users\Vinic\Downloads\archive\test"),
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
#classes (angry, surprise, happy (..))
    class_names = train_ds.class_names
    num_classes = len(class_names)
#caminho dos pesos salvos
    pesos_path = "PesosIA3.weights.h5"
    modelo_path = "ModeloIA3.h5"
#modelo
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(num_classes, activation='softmax')
    ])
#compilador adam
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


    # Modelo para inferência sem data augmentation
inference_model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(num_classes, activation='softmax')
])

# Carregar pesos treinados
inference_model.load_weights(pesos_path)



image_path = r'C:\Users\Vinic\Downloads\archive\test\sad\PrivateTest_7839088.jpg'

# 1) Abre e converte para grayscale (1 canal)
img = Image.open(image_path).convert("L").resize((img_width, img_height))

# 2) Converte para array numpy (48,48)
img_array = np.array(img, dtype=np.float32)

# 3) Expande para (48,48,1)
img_array = np.expand_dims(img_array, axis=-1)

# 4) Repete o canal para ter 3 canais (48,48,3)
img_array = np.repeat(img_array, 3, axis=-1)

# 5) Expande dimensão para batch (1,48,48,3)
img_array = np.expand_dims(img_array, axis=0)

# Agora img_array está pronta para previsão
predictions = inference_model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
print("Classe prevista:", predicted_class)

