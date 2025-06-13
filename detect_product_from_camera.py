import cv2
import numpy as np
import tensorflow as tf

# --- PATHS ---
TFLITE_PATH = "converted_tflite/model_unquant.tflite"
LABELS_PATH = "converted_tflite/labels.txt"

# --- LOAD LABELS ---
with open(LABELS_PATH, 'r') as f:
    class_names = [l.strip() for l in f if l.strip()]

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- CAPTURE IMAGE ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
print("Press SPACE to capture image...")

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Live", frame)
    if cv2.waitKey(1)&0xFF==ord(' '):
        img = frame.copy()
        break
cap.release()
cv2.destroyAllWindows()

# --- PREPROCESS ---
h, w = input_details[0]['shape'][1:3]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (w, h))

dtype = input_details[0]['dtype']
if dtype == np.float32:
    inp = np.expand_dims(resized.astype(np.float32)/255.0, axis=0)
else:
    inp = np.expand_dims(resized.astype(np.uint8), axis=0)

# --- INFERENCE ---
interpreter.set_tensor(input_details[0]['index'], inp)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])[0]

# --- RESULTS ---
for i, prob in enumerate(out):
    print(f"{class_names[i]}: {prob:.2f}")
idx = np.argmax(out)
label = class_names[idx]
conf = out[idx]
if conf>0.5:
    print(f"Detected: {label} ({conf:.2f})")
else:
    print(f"Uncertain: maybe {label} ({conf:.2f})")