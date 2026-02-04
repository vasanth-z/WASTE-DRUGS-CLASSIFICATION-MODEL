# ♻️ Waste & Drug Classification using CNN (Real-Time)
## Waste Classification ML Model

This project implements a Convolutional Neural Network (CNN) model to classify **waste** (biodegradable vs non-biodegradable) and **drugs** (legal vs illegal) from images or real-time webcam input. It uses **TensorFlow**, **OpenCV**, **Streamlit**, and **Tkinter** to enable an interactive experience for both web and desktop environments.

---

## 📌 Table of Contents:

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Tech Stack](#tech-stack)
- [Dataset Structure](#dataset-structure)
- [CNN Model Architecture](#cnn-model-architecture)
- [Training Details](#training-details)
- [User Interfaces](#user-interfaces)
  - [Tkinter](#tkinter)
  - [Streamlit](#streamlit)
- [How to Use](#how-to-use)
- [License](#license)

---

## 📖 Overview

This project classifies uploaded or real-time images into one of two categories:

1. **Waste Type Classification**:  
   - Biodegradable  
   - Non-Biodegradable  

2. **Drug Classification**:  
   - Legal Drugs (e.g., Paracetamol, Aspirin, Antibiotics)  
   - Illegal Drugs (e.g., Cocaine, Heroin, Ecstasy)  

---

## 🎯 Project Goals

- Build a **CNN model** for multi-domain classification
- Create a **Streamlit web app** for real-time predictions
- Implement **Tkinter UI** for local desktop users
- Use **OpenCV** for camera integration
- Extend the system to support **multiple classes** and **dynamic model switching**

---

## ⚙️ Tech Stack

| Technology       | Usage                                   |
|------------------|------------------------------------------|
| Python 3.x       | Programming Language                     |
| TensorFlow/Keras | Deep learning model building             |
| OpenCV           | Webcam access & image manipulation       |
| Streamlit        | Web UI for real-time predictions         |
| Tkinter          | Desktop GUI interface                    |
| PIL (Pillow)     | Image processing                         |
| NumPy & OS       | Numerical and file utilities             |

---

## 🗃️ Dataset Structure

Dataset/
├── train/

│ ├── Biodegradable/

│ ├── Non-Biodegradable/

│ ├── Legal Drugs/

│ └── Illegal Drugs/

├── val/

├── Biodegradable/

├── Non-Biodegradable/

├── Legal Drugs/

└── Illegal Drugs/



---

### 💊 Drug Dataset Folder Structure



## CNN Architecture (Same for Waste & Drugs)

python
Copy
Edit
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # For binary classification
])

- 📏 Input Shape: 128x128x3

- 📉 Loss Function: Binary Crossentropy

- 🚀 Optimizer: Adam
  ---

## 🖥️ User Interfaces
### 📌 Tkinter
Uses tkinter to display a simple GUI

Image preview and predicted label shown

Webcam input supported

### 📌 Streamlit
Web-based interface

Buttons for:

Selecting classification type (Waste / Drugs)

Uploading images

Enabling webcam for real-time predictions

## 🚀 How to Use

This project contains modules for both **waste classification** and **drug classification** using CNN and Streamlit/Tkinter interfaces. Follow the steps below to set up and run the application.

---

### 🔧 Install Dependencies

1.Clone the Repository

```bash
git clone https://github.com/vasanth-z/Garbage-ML-model.git
cd garbage-classification-cnn
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run Tkinter UI

```bash
python ui.py
```

4. Run Streamlit App

 ```bash
streamlit run streamlit.py
```
---

### 📬 Contact

VASANTH S

Email: svasanth2392005@gmail.com



---

### LICENSE

MIT License

Copyright (c) 2025 VASANTH S

Permission is hereby granted, free of charge, to any person obtaining a copy......



