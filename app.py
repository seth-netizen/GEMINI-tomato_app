import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# 1. Page Layout
st.set_page_config(page_title="AI Greenhouse Scout", layout="wide")

@st.cache_resource
def load_model():
    # Loading the model once to keep the video smooth
    return YOLO("best.pt")

model = load_model()

st.title("🍅 Greenhouse Real-Time AI Tracker")
st.markdown("### Continuous Object Tracking & Identification")

# 2. Upload Video
uploaded_video = st.file_uploader("Upload Greenhouse Walkthrough", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save video to a temporary file for processing
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    vid = cv2.VideoCapture(tfile.name)
    
    # Placeholders for the "Live Feed"
    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()  # This makes the video update in place
    metric_placeholder = col2.empty()

    # 3. The Continuous Tracking Loop
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        
        # --- THE MAGIC LINE ---
        # persist=True: Keeps the boxes "locked" on the same tomato as you move
        # tracker="bytetrack.yaml": The industry standard for car-style tracking
        results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml", verbose=False)
        
        # Get the video frame with the boxes and ID numbers drawn on it
        annotated_frame = results[0].plot() 

        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # 4. Update the UI continuously
        video_placeholder.image(frame_rgb, use_column_width=True)
        
        # Update the live count on the side
        if results[0].boxes.id is not None:
            current_count = len(results[0].boxes.id)
            metric_placeholder.metric("📍 Current Targets in View", current_count)

    vid.release()
    st.success("✅ Analysis Complete!")

else:
    st.info("Awaiting video upload to begin live tracking...")
