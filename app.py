import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import pandas as pd

# 1. Page Setup
st.set_page_config(page_title="AI Greenhouse Scout", page_icon="🍅", layout="wide")

# Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ff4b4b; }
    .stTable { background-color: #161b22; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Model Loading (Local Windows Path)
# Using raw string (r"") to handle Windows backslashes correctly
# Change this line in your refined app.py
MODEL_PATH = "best.pt"


@st.cache_resource
def load_tomato_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_tomato_model(MODEL_PATH)

# --- HEADER ---
st.title("🍅 Greenhouse Live AI Scout")
if model:
    st.sidebar.success("✅ Model Loaded Successfully")
else:
    st.sidebar.error("❌ Model not found at path. Check the path in the code.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Detection Settings")
conf_val = st.sidebar.slider("Confidence (Sensitivity)", 0.1, 1.0, 0.35)
iou_val = st.sidebar.slider("Overlap (IOU) Threshold", 0.1, 1.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.info("As you move through the greenhouse, this AI tracks and counts each unique tomato.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload Greenhouse Walkthrough Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save video to temp file for processing
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # UI Layout: Video on left, Stats on right
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Live Analytics")
        total_metric = st.empty()
        table_placeholder = st.empty()

    # --- PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use .track() for moving video to prevent double-counting same objects
        results = model.track(frame, conf=conf_val, iou=iou_val, persist=True, verbose=False)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Calculate counts from the current frame
        class_names = results[0].names
        counts = {}
        if results[0].boxes.cls is not None:
            classes_detected = results[0].boxes.cls.cpu().numpy()
            for cls_id in classes_detected:
                label = class_names[int(cls_id)]
                counts[label] = counts.get(label, 0) + 1

        # --- UPDATE UI ---
        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_column_width=True)
        
        # Update Metrics
        total_count = sum(counts.values())
        total_metric.metric("Total Objects Seen", total_count)
        
        if counts:
            df = pd.DataFrame(list(counts.items()), columns=['Class', 'Count'])
            table_placeholder.table(df)

    cap.release()
    st.balloons() # Visual celebration when video finishes
    st.success("Analysis Complete!")

else:
    st.info("Awaiting video upload to begin scouting...")
