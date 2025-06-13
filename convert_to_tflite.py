import tensorflow as tf
import os

KERAS_MODEL = "keras_model.h5"
TFLITE_MODEL = "converted_tflite/model_unquant.tflite"

# load keras
model = tf.keras.models.load_model(KERAS_MODEL)
# converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# (optional) quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_bytes = converter.convert()

os.makedirs(os.path.dirname(TFLITE_MODEL), exist_ok=True)
with open(TFLITE_MODEL, 'wb') as f:
    f.write(tflite_bytes)
print(f"✅ TFLite model saved to {TFLITE_MODEL}")