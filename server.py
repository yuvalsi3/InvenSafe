from __future__ import annotations

from flask import Flask, request
from flask_cors import CORS
import subprocess
import json
import os
import sys
from pathlib import Path

# ── Cleanup on startup ─────────────────────────────────────────────────────────
def clear_face_data_folder():
    folder = "face_data"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"🧹 Deleted file: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
            else:
                print(f"ℹ️ Skipped (not a file): {file_path}")

# Call it immediately on startup
clear_face_data_folder()

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Dynamic paths (no more hard-coded absolute strings) ────────────────────────
HERE = Path(__file__).resolve().parent          # directory that contains this file
PYTHON_EXE = sys.executable                     # the Python that runs Flask

@app.route("/capture", methods=["POST"])
def capture():
    data = request.json
    transaction_id = data.get("transaction_id", "unknown")
    items = data.get("items", [])

    os.makedirs("face_data", exist_ok=True)
    with open(f"face_data/{transaction_id}_products.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    try:
        subprocess.Popen([
            PYTHON_EXE,
            str(HERE / "trigger_only.py"),
            transaction_id
        ])
    except Exception as e:
        print(f"❌ שגיאה בהפעלת trigger: {e}")

    return "📸 נרשם בהצלחה"

@app.route("/trigger_status", methods=["GET"])
def trigger_status():
    transaction_id = request.args.get("transaction_id")
    if not transaction_id:
        return "❌ חסר transaction_id"

    done_file = f"face_data/{transaction_id}_trigger_done.txt"
    if os.path.exists(done_file):
        return "TRIGGER_DONE"
    return "WAITING"

@app.route("/verify", methods=["POST"])
def verify():
    data = request.json
    transaction_id = data.get("transaction_id", "unknown")

    try:
        result = subprocess.run([
            PYTHON_EXE,
            str(HERE / "check_fridge_user.py")
        ], capture_output=True, text=True, encoding="utf-8", errors="ignore")

        output = result.stdout
        print(output)

        if "MATCH_FOUND" in output:
            print("✅ פנים זוהו – המנעול נפתח")
            return "MATCH_FOUND"
        elif "NO_MATCH" in output:
            return "NO_MATCH"
        else:
            return output

    except Exception as e:
        print(f"❌ שגיאה באימות: {e}")
        return "🚫 שגיאה באימות"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        transaction_id = data.get("transaction_id")
        if not transaction_id:
            return "❌ חסר transaction_id ב-body של הבקשה"

        result = subprocess.run([
            PYTHON_EXE,
            str(HERE / "predict.py"),
            transaction_id
        ], capture_output=True, text=True, encoding="utf-8", errors="ignore")

        print(result.stdout)
        return result.stdout

    except Exception as e:
        print(f"❌ שגיאה בהרצת predict: {e}")
        return "🚫 שגיאה בזיהוי המוצר"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


    # sb-fjmtp43241818@personal.example.com
    # B?3hGm$5