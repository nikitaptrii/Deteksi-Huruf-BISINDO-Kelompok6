import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("bisindo_model.h5")

# UI Setup
st.set_page_config(layout="centered")
st.title(" 🤟 Deteksi Huruf BISINDO 🔠 ")
st.markdown("📷 Aktifkan kamera untuk mulai mendeteksi huruf dengan bahasa isyarat BISINDO.")

# Checkbox untuk mengaktifkan kamera
run = st.checkbox("🎥 Aktifkan Kamera")
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap is not None:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠ Tidak dapat membaca dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Kotak ROI (Region of Interest) di tengah
    box_size = 300
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    roi = frame[y1:y2, x1:x2]

    # Preprocessing untuk model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 1))

    # Prediksi huruf
    prediction = model.predict(reshaped, verbose=0)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index] * 100
    letter = chr(65 + class_index)
    pred_text = f"{letter} ({confidence:.1f}%)"

    # Tambahkan teks dan kotak prediksi ke frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
    cv2.putText(frame, f"Prediksi: {pred_text}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 3)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb, caption=f"Prediksi: {pred_text}")

# Stop webcam jika selesai
if cap:
    cap.release()