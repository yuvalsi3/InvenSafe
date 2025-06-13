import sys
import cv2
import os
import json
from deepface import DeepFace

def trigger_photo(transaction_id):
    print("\n========================")
    print("📸 מופעל trigger_photo")
    print("========================")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ המצלמה לא נפתחה!")
        return

    found_face = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("📷 מצלמה", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            os.makedirs("face_data", exist_ok=True)
            filename = f"face_data/{transaction_id}_photo.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ נשמרה תמונה: {filename}")
            with open(f"face_data/{transaction_id}_trigger_done.txt", "w") as f:
                f.write("TRIGGER_DONE")

            # חילוץ embedding
            try:
                embedding_obj = DeepFace.represent(img_path=filename, model_name='Facenet')[0]
                embedding = embedding_obj["embedding"]

                with open(f"face_data/{transaction_id}_embedding.json", "w") as f:
                    json.dump(embedding, f)
                print("✅ נשמר embedding")
                found_face = True
            except Exception as e:
                print(f"❌ שגיאה בזיהוי פנים: {e}")

            break

    if not found_face:
        print("🤔 לא זוהו פנים")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trigger_photo(sys.argv[1])
    else:
        print("❗ נא להזין transaction_id בעת ההפעלה")