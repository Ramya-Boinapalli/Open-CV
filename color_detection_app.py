# Mediapipe_app.py
import streamlit as st
import cv2
import numpy as np

st.title("HSV Color Detection App")

# Checkbox to turn webcam on/off
run = st.checkbox('Run Webcam')

# Placeholder for video frames
frame_window = st.image([])

# HSV color ranges
COLOR_RANGES = {
    "Red": (np.array([161, 155, 84]), np.array([179, 255, 255])),
    "Blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
    "Green": (np.array([40, 100, 100]), np.array([102, 255, 255])),
}

# HSV range for "all colors except white"
ALL_EXCEPT_WHITE = (np.array([0, 42, 0]), np.array([179, 255, 255]))

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color masks
        masks = {}
        for color, (low, high) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv_frame, low, high)
            masks[color] = cv2.bitwise_and(frame, frame, mask=mask)

        # Mask for all colors except white
        mask_all = cv2.inRange(hsv_frame, ALL_EXCEPT_WHITE[0], ALL_EXCEPT_WHITE[1])
        result = cv2.bitwise_and(frame, frame, mask=mask_all)

        # Stack images horizontally
        combined_image = np.hstack((
            cv2.resize(frame, (320, 240)),
            cv2.resize(masks["Red"], (320, 240)),
            cv2.resize(masks["Blue"], (320, 240)),
            cv2.resize(masks["Green"], (320, 240)),
            cv2.resize(result, (320, 240))
        ))

        # Convert BGR → RGB for Streamlit
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        frame_window.image(combined_image)

    cap.release()
    cv2.destroyAllWindows()
