import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


st.title("♻️ Drug & Waste Classification")
st.markdown("Choose a classification type, upload an image, or use your webcam for real-time classification.")


classification_type = st.radio("Select Classification Type", ["Drug Classification", "Waste Classification"])


if classification_type == "Drug Classification":
    model_path = "waste_drugs_classification.h5"
    class_labels = ["legal Drugs", "Illegal Drugs"]
elif classification_type == "Waste Classification":
    model_path = "waste_classification.h5"
    class_labels = ["Non-Biodegradable", "Biodegradable"]


model = tf.keras.models.load_model(model_path)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


camera = st.checkbox("Use Webcam")



def classify_image(img):
    img = img.resize((128, 128))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0) 

    prediction = model.predict(img)  

    if classification_type == "Waste Classification":
        predicted_class = class_labels[int(prediction[0] > 0.5)]  
    else:
        predicted_class_index = np.argmax(prediction)  
        predicted_class = class_labels[predicted_class_index]  

    return predicted_class



if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = classify_image(image)
    st.success(f"Prediction: *{result}*")


if camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = classify_image(img)

        
        cv2.putText(frame, f"Prediction: {result}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
