import cv2
import os
import json
import numpy as np
import tensorflow as tf
import time
import datetime
import sys

# קריאת transaction_id מהפרמטרים
if len(sys.argv) < 2:
    print("❌ לא הועבר transaction_id כפרמטר")
    exit()

transaction_id = sys.argv[1]
print(f"🔁 קיבלתי Transaction ID: {transaction_id}")

# קריאת מוצרים ששולמו
def load_paid_products(transaction_id):
    path = f"face_data/{transaction_id}_products.json"
    if not os.path.exists(path):
        print(f"❌ הקובץ {path} לא נמצא")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [item["name"] for item in json.load(f)]

# הגדרת מודל
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# טעינת תוויות
with open("converted_tflite/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# פתיחת מצלמה
cap = cv2.VideoCapture(2)  # שנה ל-1,2,3 אם צריך
if not cap.isOpened():
    print("❌ לא נפתחה מצלמה")
    exit()

# תיקיית תמונות
image_dir = f"face_data/products_{transaction_id}"
os.makedirs(image_dir, exist_ok=True)

# מוצרים ששולמו
paid_names = load_paid_products(transaction_id)
if not paid_names:
    print("❌ אין מוצרים בתשלום. יציאה.")
    cap.release()
    exit()

recognized_products = []
paid_names_copy = paid_names.copy()

print("⌛ ממתין 5 שניות לפני תחילת הצילום...")
time.sleep(5)

# צילום לכל מוצר
for i, product in enumerate(paid_names, start=1):
    print(f"\n📸 מצלמים מוצר מספר {i}: {product}")

    start_time = time.time()
    product_found = None

    while time.time() - start_time < 10:  # 10 שניות לכל מוצר
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("📷 Product Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_dir}/product_{i}_{timestamp}.jpg"
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"✅ נשמרה תמונה: {filename}")
        else:
            print("❌ כשל בשמירת תמונה")
        break  # צילום פעם אחת

    # עיבוד לזיהוי
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data)
    predicted_class = class_names[predicted_index]
    confidence = float(output_data[0][predicted_index])

    print(f"🎯 זוהה: {predicted_class} (ביטחון: {confidence:.2f})")

    if confidence > 0.7:
        recognized_products.append(predicted_class)
        if predicted_class in paid_names_copy:
            paid_names_copy.remove(predicted_class)
        else:
            print(f"❌ מוצר לא נרכש: {predicted_class}")
    else:
        print("⚠️ לא זוהה מוצר")
        recognized_products.append("לא זוהה")

cap.release()
cv2.destroyAllWindows()

# שמירת תוצאה
result_path = f"{image_dir}/result.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("🧾 מוצרים בתשלום:\n")
    f.write(json.dumps(paid_names, ensure_ascii=False, indent=2) + "\n\n")
    f.write("📦 מוצרים שזוהו בפועל:\n")
    f.write(json.dumps(recognized_products, ensure_ascii=False, indent=2) + "\n\n")

    if not paid_names_copy:
        success_message = "✅ כל המוצרים נלקחו בהצלחה!"
        f.write(success_message + "\n")
        print(success_message)
    else:
        missing_message = f"⚠️ חסרים מוצרים: {paid_names_copy}"
        f.write(missing_message + "\n")
        print(missing_message)

print(f"\n📁 כל התמונות והתוצאה נשמרו בתיקייה: {image_dir}")