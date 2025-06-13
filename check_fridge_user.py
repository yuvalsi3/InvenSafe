import cv2
import os
import json
import numpy as np
from deepface import DeepFace

# טעינת כל המאגר
def load_all_embeddings(folder_path):
    faces = {}
    for file in os.listdir(folder_path):
        if file.endswith("_embedding.json"):
            with open(os.path.join(folder_path, file), "r") as f:
                data = json.load(f)
                faces[file] = np.array(data)
    return faces

saved_faces = load_all_embeddings("face_data")
if not saved_faces:
    print("❌ אין נתוני פנים במאגר")
    exit()

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠️ לא נפתחה מצלמה")
    exit()

found_face = False
match_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("📷 מצלמה", frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        try:
            embedding_obj = DeepFace.represent(frame, model_name='Facenet')[0]
            new_embedding = np.array(embedding_obj["embedding"])

            best_match = None
            min_dist = float('inf')

            for name, saved_embedding in saved_faces.items():
                dist = np.linalg.norm(new_embedding - saved_embedding)
                print(f"🔍 בדיקה מול {name} | מרחק: {dist:.4f}")
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

            if min_dist < 9:  # הסף הזה עובד מצוין ל-Facenet
                print(f"✅ נמצאה התאמה: {best_match} | מרחק: {min_dist:.4f}")
                match_found = True
            else:
                print(f"❌ אין התאמה | מרחק: {min_dist:.4f}")

        except Exception as e:
            print(f"❌ שגיאה בזיהוי פנים: {e}")
        break

cap.release()
cv2.destroyAllWindows()

if match_found:
    print("MATCH_FOUND")
else:
    print("NO_MATCH")