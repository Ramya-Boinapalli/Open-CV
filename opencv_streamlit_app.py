import streamlit as st
import cv2
import numpy as np
import tempfile
import time

st.set_page_config(page_title="OpenCV Demo App", layout="wide")
st.title("🖼️ OpenCV Real-Time Processing Dashboard")
st.markdown("Interactive demo for color detection, flipping, grayscale, colormaps, and video/image processing")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
use_webcam = st.sidebar.checkbox("Use Webcam", True)
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg","jpeg","png","mp4"])
flip_horizontal = st.sidebar.checkbox("Flip Horizontally", False)
flip_vertical = st.sidebar.checkbox("Flip Vertically", False)
show_gray = st.sidebar.checkbox("Show Grayscale", True)
colormap_option = st.sidebar.selectbox("Colormap", ["None", "JET", "HOT", "COOL"])
run_btn = st.sidebar.button("🚀 Run Demo")

# Helper functions
def to_rgb(frame):
    """Convert any image to RGB for Streamlit display."""
    if len(frame.shape) == 2:  # grayscale
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:    # RGBA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def apply_colormap(frame, cmap_name):
    """Apply OpenCV colormap if selected."""
    if cmap_name == "JET":
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    elif cmap_name == "HOT":
        return cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    elif cmap_name == "COOL":
        return cv2.applyColorMap(frame, cv2.COLORMAP_COOL)
    return frame

def process_frame(frame):
    """Generate masks for Red, Blue, Green, and All Except White."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {}
    
    red_mask = cv2.inRange(hsv, np.array([161,155,84]), np.array([179,255,255]))
    masks['Red'] = cv2.bitwise_and(frame, frame, mask=red_mask)
    
    blue_mask = cv2.inRange(hsv, np.array([94,80,2]), np.array([126,255,255]))
    masks['Blue'] = cv2.bitwise_and(frame, frame, mask=blue_mask)
    
    green_mask = cv2.inRange(hsv, np.array([40,100,100]), np.array([102,255,255]))
    masks['Green'] = cv2.bitwise_and(frame, frame, mask=green_mask)
    
    all_mask = cv2.inRange(hsv, np.array([0,42,0]), np.array([179,255,255]))
    masks['All Except White'] = cv2.bitwise_and(frame, frame, mask=all_mask)
    
    # Ensure all masks are 3-channel BGR
    for key in masks:
        if len(masks[key].shape) == 2:
            masks[key] = cv2.cvtColor(masks[key], cv2.COLOR_GRAY2BGR)
    
    if flip_horizontal:
        for key in masks:
            masks[key] = cv2.flip(masks[key], 1)
    if flip_vertical:
        for key in masks:
            masks[key] = cv2.flip(masks[key], 0)
    return masks

def stream_frames(cap):
    stframe_orig = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Flip original frame
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        if flip_vertical:
            frame = cv2.flip(frame, 0)

        masks = process_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if show_gray else None
        colored = apply_colormap(frame, colormap_option)

        # Tabs for organized display
        tabs = st.tabs(["Original", "Color Masks", "Grayscale", "Colormap"])
        
        with tabs[0]:
            st.image(to_rgb(frame), caption="Original Frame", use_container_width=True)
        
        with tabs[1]:
            cols = st.columns(4)
            cols[0].image(to_rgb(masks['Red']), caption="Red", use_container_width=True)
            cols[1].image(to_rgb(masks['Blue']), caption="Blue", use_container_width=True)
            cols[2].image(to_rgb(masks['Green']), caption="Green", use_container_width=True)
            cols[3].image(to_rgb(masks['All Except White']), caption="All Except White", use_container_width=True)
        
        with tabs[2]:
            if gray is not None:
                st.image(to_rgb(gray), caption="Grayscale", use_container_width=True)
        
        with tabs[3]:
            if colormap_option != "None":
                st.image(to_rgb(colored), caption=f"Colormap: {colormap_option}", use_container_width=True)

        time.sleep(0.03)

# Run the app
if run_btn:
    if use_webcam:
        cap = cv2.VideoCapture(0)
        stream_frames(cap)
        cap.release()
    elif uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)
        stream_frames(cap)
        cap.release()
    else:
        st.warning("Upload a video/image or enable webcam to start.")
