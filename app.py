import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the AI Brain
model = YOLO('best.pt') 

st.title("🍅 AI Greenhouse Scout: Live Video Mode")
st.write("Point your camera at the tomatoes to start real-time detection.")

# Setup for counts
if 'counts' not in st.session_state:
    st.session_state.counts = {"bloom": 0, "green": 0, "turning": 0, "red": 0, "damaged": 0}

# Camera Input
run_video = st.toggle('Start Greenhouse Camera')
FRAME_WINDOW = st.image([]) # This will hold the video feed

cam = cv2.VideoCapture(0) # 0 is the default camera

while run_video:
    ret, frame = cam.read()
    if not ret:
        st.error("Camera not found.")
        break

    # 1. Physics & AI: Run YOLO on the current frame
    results = model.predict(frame, conf=0.5) # conf=0.5 for stability
    
    # 2. Draw boxes on the frame
    annotated_frame = results[0].plot()
    
    # 3. Math: Calculate current counts
    class_ids = results[0].boxes.cls.astype(int).tolist()
    names = results[0].names
    current_counts = {names[i]: class_ids.count(i) for i in set(class_ids)}
    
    # Update display
    FRAME_WINDOW.image(annotated_frame, channels="BGR")
    
    # 4. Display Live Metrics
    st.subheader("Live Yield Count")
    cols = st.columns(len(st.session_state.counts))
    for i, (label, _) in enumerate(st.session_state.counts.items()):
        val = current_counts.get(label, 0)
        cols[i].metric(label.upper(), val)
    
    total = sum(current_counts.values())
    st.markdown(f"### **Total Detected: {total}**")

else:
    st.info("Camera is off. Toggle the switch above to start.")