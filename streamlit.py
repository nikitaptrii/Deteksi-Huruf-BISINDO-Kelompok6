import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("bisindo_model.h5")

st.title("Deteksi Huruf BISINDO")
st.markdown("Klik tombol untuk mengaktifkan / menonaktifkan kamera, lalu ambil gambar.")

# State kontrol kamera
if "kamera_aktif" not in st.session_state:
    st.session_state.kamera_aktif = False

# Tombol Start/Stop Kamera
if not st.session_state.kamera_aktif:
    if st.button("🎥 Mulai Kamera"):
        st.session_state.kamera_aktif = True
else:
    if st.button("🛑 Stop Kamera"):
        st.session_state.kamera_aktif = False

# Placeholder untuk gambar
FRAME_WINDOW = st.empty()

# Hanya aktif jika kamera dinyalakan
if st.session_state.kamera_aktif:
    cap = cv2.VideoCapture(0)
    st.info("Kamera aktif. Tekan tombol Ambil Gambar di bawah.")

    if st.button("📸 Ambil Gambar"):
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            box = 300
            x1 = w // 2 - box // 2
            y1 = h // 2 - box // 2
            x2 = x1 + box
            y2 = y1 + box

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

            # Tambah kotak dan teks prediksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Prediksi: {text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=text)
        else:
            st.error("Gagal menangkap gambar dari kamera.")
else:
    st.warning("Kamera belum dinyalakan.")