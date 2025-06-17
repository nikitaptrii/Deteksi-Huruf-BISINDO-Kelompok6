import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('bisindo_model.h5')

st.title("Deteksi Huruf BISINDO")
st.markdown("Gunakan kamera untuk mendeteksi huruf BISINDO secara real-time.")

# Buat state untuk tracking status kamera
if "kamera_jalan" not in st.session_state:
    st.session_state.kamera_jalan = False

# Tombol mulai / stop
if not st.session_state.kamera_jalan:
    if st.button("🎥 Mulai Kamera", key="mulai"):
        st.session_state.kamera_jalan = True
else:
    if st.button("🛑 Stop Kamera", key="stop"):
        st.session_state.kamera_jalan = False

# Tampilkan video jika kamera aktif
frame_placeholder = st.empty()

if st.session_state.kamera_jalan:
    cap = cv2.VideoCapture(0)

    while st.session_state.kamera_jalan:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        box_size = 300
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 64, 64, 1))

        prediction = model.predict(reshaped, verbose=0)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index] * 100
        letter = chr(65 + class_index)

        if confidence > 80:
            text = f"{letter} ({confidence:.1f}%)"
        else:
            text = "Tidak yakin"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Prediksi: {text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)

        # Periksa status kamera
        if not st.session_state.kamera_jalan:
            break

    cap.release()
