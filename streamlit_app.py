import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("bisindo_model.h5")

# Streamlit UI setup
st.title("BISINDO Real-time Webcam Recognition")

# Start/stop webcam based on user input
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

# Start webcam if 'Start Webcam' is checked
while run and cap is not None:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI) for detection
    roi = frame[100:300, 100:300]  # You can adjust this as needed

    # Convert ROI to grayscale and preprocess it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Resize to match input size of the model
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 64, 64, 1))  # Reshape to model input format

    # Predict the letter
    prediction = model.predict(reshaped)
    letter = chr(np.argmax(prediction) + 65)  # Convert prediction to letter A-Z

    # Draw rectangle around the ROI and display prediction
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediksi: {letter}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame from BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in the Streamlit window
    FRAME_WINDOW.image(frame)

# Release webcam when done
if cap:
    cap.release()
