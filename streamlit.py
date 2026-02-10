import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("waste_classification.h5")

# Define class labels
class_labels = ["Non-Biodegradable", "Biodegradable"]

# Streamlit UI
st.title("♻️ Waste Classification - Biodegradable vs Non-Biodegradable")
st.markdown("Upload an image or use your webcam for real-time classification.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# OpenCV Camera Feed
camera = st.checkbox("Use Webcam")

# Image Processing & Prediction Function
def classify_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions

    prediction = model.predict(img)
    predicted_class = class_labels[int(prediction[0] > 0.5)]
    return predicted_class

# Process Uploaded Image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = classify_image(image)
    st.success(f"Prediction: *{result}*")

# Process Webcam Stream
if camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert to PIL for model processing
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = classify_image(img)

        # Draw prediction text on frame
        cv2.putText(frame, f"Prediction: {result}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display frame
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
