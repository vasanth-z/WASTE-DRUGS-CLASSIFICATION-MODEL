# # # Import necessary libraries
# # import cv2
# # import numpy as np
# # import tensorflow as tf

# # # Load trained model
# # model = tf.keras.models.load_model("waste_classification.h5")

# # # Define class labels
# # class_labels = ["Non-Biodegradable", "Biodegradable"]

# # # Initialize webcam
# # cap = cv2.VideoCapture(1)  # 0 is the default camera

# # while True:
# #     ret, frame = cap.read()  # Capture frame from webcam
# #     if not ret:
# #         break  # Stop if no frame is captured

# #     # Preprocess the frame
# #     img = cv2.resize(frame, (128, 128))  # Resize to match training size
# #     img = img / 255.0  # Normalize pixel values
# #     img = np.expand_dims(img, axis=0)  # Add batch dimension

# #     # Get prediction
# #     prediction = model.predict(img)
# #     predicted_class = class_labels[int(prediction[0] > 0.5)]  # Binary classification

# #     # Display prediction on frame
# #     cv2.putText(frame, f"Prediction: {predicted_class}", (20, 50),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# #     # Show the webcam feed
# #     cv2.imshow("Waste Classification", frame)

# #     # Press 'q' to exit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()




import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# Load trained model
model = tf.keras.models.load_model("waste_classification.h5")

# Define class labels
class_labels = ["Non-Biodegradable", "Biodegradable"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create the UI window
root = tk.Tk()
root.title("Waste Classification")

# Create a label to display the webcam feed
video_label = tk.Label(root)
video_label.pack()

# Create a label for prediction result
prediction_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
prediction_label.pack()


def update_frame():
    """Capture frame, classify, and update UI"""
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Preprocess the frame
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get prediction
    prediction = model.predict(img)
    predicted_class = class_labels[int(prediction[0] > 0.5)]

    # Convert frame to RGB (Tkinter does not support BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update UI elements
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    prediction_label.config(text=f"Prediction: {predicted_class}")

    # Repeat function every 10ms
    root.after(10, update_frame)


# Start updating the UI
update_frame()
root.mainloop()

# Release resources when window is closed
cap.release()




