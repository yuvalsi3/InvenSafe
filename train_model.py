import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- PARAMETERS ---
DATA_DIR = "real_drink_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
OUTPUT_MODEL = "keras_model.h5"
OUTPUT_LABELS = "converted_tflite/labels.txt"

# --- DATA AUGMENTATION & LOADING ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    zoom_range=0.2,
    rotation_range=10
)
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
num_classes = train_gen.num_classes

# --- BUILD MODEL ---
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(128,3, activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'), layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

# --- TRAIN ---
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --- SAVE KERAS MODEL & LABELS ---
model.save(OUTPUT_MODEL)

os.makedirs(os.path.dirname(OUTPUT_LABELS), exist_ok=True)
# write labels.txt in order of indices
label_map = train_gen.class_indices  # e.g. {'Coke':0, 'Fanta':1, ...}
sorted_labels = sorted(label_map.items(), key=lambda kv: kv[1])
with open(OUTPUT_LABELS, 'w') as f:
    for label, _ in sorted_labels:
        f.write(label + "\n")
print(f"✅ Training finished. Model saved to {OUTPUT_MODEL}, labels to {OUTPUT_LABELS}")