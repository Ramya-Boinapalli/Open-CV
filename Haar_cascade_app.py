# haar_cascade_app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

st.set_page_config(page_title="Haar Cascade Detection App", layout="wide")
st.title("📸 Haar Cascade Detection App")
st.write("Upload an image or video and select the detection type!")

# Sidebar options
detection_type = st.sidebar.selectbox(
    "Choose Detection Type",
    [
        "Face Detection",
        "Face & Eye Detection",
        "BGR to Gray",
        "BGR to HSV",
        "Car Detection",
        "Pedestrian Detection",
        "Sketch Transform",
        "Webcam Face Detection"
    ]
)

uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Helper functions
def load_image(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    return tfile.name

# Haar cascades paths
face_cascade_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\open_cv\Haar_cascade_classification\haarcascade_frontalface_default.xml"
eye_cascade_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\open_cv\Haar_cascade_classification\haarcascade_eye.xml"
car_cascade_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\open_cv\Haar_cascade_classification\haarcascade_frontalcatface.xml"
body_cascade_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\open_cv\Haar_cascade_classification\haarcascade_fullbody.xml"

# Load classifiers
face_classifier = cv2.CascadeClassifier(face_cascade_path)
eye_classifier = cv2.CascadeClassifier(eye_cascade_path)
car_classifier = cv2.CascadeClassifier(car_cascade_path)
body_classifier = cv2.CascadeClassifier(body_cascade_path)

# Detection functions
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
    return image

def detect_faces_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
    return image

def convert_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def detect_cars(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray, 1.1, 3)
        for (x,y,w,h) in cars:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)
    cap.release()

def detect_pedestrians(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)
    cap.release()

def sketch_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 10, 80)
    _, mask = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def webcam_face_detection():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_faces(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if st.button("Stop Webcam"):
            break
    cap.release()

# Main logic
if detection_type == "Webcam Face Detection":
    st.warning("Starting webcam detection. Click Stop Webcam to end.")
    webcam_face_detection()
elif uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext in ["jpg", "jpeg", "png"]:
        image = load_image(uploaded_file)
        if detection_type == "Face Detection":
            result = detect_faces(image)
        elif detection_type == "Face & Eye Detection":
            result = detect_faces_eyes(image)
        elif detection_type == "BGR to Gray":
            result = convert_gray(image)
        elif detection_type == "BGR to HSV":
            result = convert_hsv(image)
        elif detection_type == "Sketch Transform":
            result = sketch_transform(image)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    elif file_ext in ["mp4", "avi"]:
        video_path = load_video(uploaded_file)
        if detection_type == "Car Detection":
            detect_cars(video_path)
        elif detection_type == "Pedestrian Detection":
            detect_pedestrians(video_path)
else:
    st.warning("Please upload an image or video file.")
