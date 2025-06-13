---

### ✅ `README.md` ### 

---

## 🏪 InvenSAFE – Smart Fridge Security System

---

```
InvenSAFE is your fridge's personal bodyguard -
guarding goodies and making sure only the right faces grab the right snacks

```
## 📌 Features

```bash

- 🧠 Face recognition (DeepFace + OpenCV)
- 💳 Payment simulation via PayPal (HTML UI)
- 🔐 Access control with facial verification
- 🧾 Product detection using TensorFlow Lite
- 🎥 Live camera capture and photo storage
- 📂 Logging of transactions and audit trail
- 🌐 Python Flask backend with RESTful endpoints

```

## 📁 Project Structure

```bash
invensafe/
├── face_data/                  # Stores captured user faces, embeddings & transactions
├── real_drink_dataset/         # Raw dataset for product recognition training
├── converted_tflite/           # Labels for TensorFlow Lite model
├── dataset/                    # Image folders used during training
│
├── invensafe.html              # Frontend for product selection & payment
├── server.py                   # Flask backend
├── trigger_only.py             # Captures user photo after payment
├── check_fridge_user.py        # Verifies user face
├── predict.py                  # Runs product recognition after access
├── train-model.py              # Script to train a custom Keras product recognition model
├── keras_model.h5              # Original product recognition model
├── model_unquant.tflite        # TFLite converted model for edge deployment
├── detect_product_from_camera.py # Standalone script for product prediction
├── convert_to_tflite.py        # Script to convert .h5 → .tflite
```

---

## 🚀 Getting Started

### 1. Clone the project

```bash
git clone https://github.com/yuvalsi3/InvenSafe.git
```

### 2. Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, use the list below manually.

---

## 📦 Python Package Requirements

```bash
Flask
flask-cors
opencv-python
tensorflow
mediapipe
numpy
deepface
```

You can install all with:

```bash
pip install flask flask-cors opencv-python tensorflow mediapipe numpy deepface
```

---

## ▶️ How It Works
```
1. User selects products and pays via PayPal (UI in `invensafe.html`)
2. System captures a photo and saves it to `face_data/`
3. User tries to open fridge → system verifies their face
4. If face is matched, camera activates to detect each product
5. Products taken are compared to those paid for
6. Final result is logged and can trigger alerts if mismatch is detected
```
---

## 🧪 Training a Custom Product Model
Want to improve or replace the default product-recognition model?
Train your own in just two steps:


#### 1. Prepare your dataset
 Create a folder per product class inside dataset/, placing images in each:

```bash
dataset/
├── coke/
│   ├── img1.jpg
│   ├── img2.jpg
├── water/
│   ├── img1.jpg
│   ├── img2.jpg
```


#### 2. Run the training script


```bash
bashpython train-model.py
```

### What happens:
```
Images are loaded with ImageDataGenerator.

A simple CNN is trained and saved as keras_model.h5.

Class labels are written to converted_tflite/labels.txt.

```
Need TensorFlow Lite?
Convert the model with:


```bash 
python convert_to_tflite.py
```

The resulting model_unquant.tflite runs efficiently on edge devices (e.g., Raspberry Pi).



## 📷 Camera Note

The code uses OpenCV (`cv2.VideoCapture(index)`) – default index is `0`.

If you're using multiple webcams or getting a blank window, try:
```python
cv2.VideoCapture(1)
```

---

## 📁 Output and Logs

- Captured faces & embeddings: `face_data/`
- Product images taken during prediction: `face_data/products_<transaction_id>/`
- Verification results: `result.txt` inside the transaction folder

---

## 💬 Developer Alerts

> [!IMPORTANT]
> This project involves real-time facial recognition and camera access — ensure you're complying with privacy and data protection guidelines during usage.

> [!TIP]
> You can fine-tune detection thresholds (face or product) in the respective Python scripts to improve accuracy.

> [!NOTE]
> Make sure your webcam is accessible and not used by another app before launching `predict.py` or `check_fridge_user.py`.

## 💡 Tips

- The face verification threshold and product model confidence can be adjusted in the Python scripts.
- If your webcam is slow to start, increase the sleep delay in `predict.py`.

---

## ✨ Team & Acknowledgments

Built by **Team InvenSAFE**  
Academic College of Tel Aviv-Yaffo | Practical Learning Accelerator B  
Supervised by: Ron Rozen

---

## 🛠️ Some Tools I Used and Learned

<p align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="python" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg" alt="flask" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" alt="opencv" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" alt="tensorflow" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" alt="html5" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" alt="javascript" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" alt="git" width="45" height="45"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="github" width="45" height="45"/>
</p>

---

## 📃 License

This project is licensed under the MIT License.
```
Let me know if you'd also like:
- A matching `requirements.txt`
- `instructions.txt` for end-users (non-devs)
- README in Hebrew

I'm happy to help!
```
