import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
import tempfile

# 1. Page Config for a Pro "AI Dashboard" look
st.set_page_config(page_title="Greenhouse Scout AI", page_icon="🍅", layout="wide")

# Custom CSS for the "Wow" factor in presentations
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ff4b4b; font-size: 28px; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Optimized Model Loader
@st.cache_resource
def load_tomato_model():
    return YOLO("best.pt")

model = load_tomato_model()

# --- SIDEBAR: Controls & Settings ---
st.sidebar.image("https://icons8.com", width=80)
st.sidebar.title("🛠️ AI Control Panel")

mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["📁 Upload Image", "🎥 Upload Video", "📸 Live Camera"]
)

confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use lower confidence for smaller bloom flowers.")

# --- MAIN INTERFACE ---
st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.markdown("### Real-time Yield Analysis & Ripeness Tracking")

# Create placeholders for a clean layout
col_left, col_right = st.columns([2, 1])

source_data = None

# --- INPUT LOGIC ---
if mode == "📸 Live Camera":
    source_data = st.camera_input("Capture Greenhouse Frame")
elif mode == "📁 Upload Image":
    source_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif mode == "🎥 Upload Video":
    source_data = st.file_uploader("Upload Greenhouse Walkthrough Video", type=["mp4", "mov", "avi"])

# --- PROCESSING LOGIC ---
if source_data:
    # --- VIDEO PROCESSING ---
    if mode == "🎥 Upload Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_data.read())
        vid = cv2.VideoCapture(tfile.name)
        
        with col_left:
            st_frame = st.empty()
        
        with col_right:
            st.subheader("📊 Live Tracking")
            metric_placeholder = st.empty()
            chart_placeholder = st.empty()

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret: break
            
            # Use .track() for video to prevent flicker
            results = model.track(frame, conf=confidence, persist=True, verbose=False)
            annotated_frame = results[0].plot()
            
            # Update Video
            st_frame.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Update Stats
            if results[0].boxes.id is not None:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                counts = Counter(classes)
                names = results[0].names
                
                # Show top 3 detections as metrics
                with metric_placeholder.container():
                    m_cols = st.columns(2)
                    for i, (cls_id, count) in enumerate(list(counts.items())[:4]):
                        m_cols[i % 2].metric(names[cls_id].upper(), count)
        vid.release()

    # --- IMAGE PROCESSING (Camera or Upload) ---
    else:
        image = Image.open(source_data)
        img_array = np.array(image)

        with st.spinner("🔍 Analyzing Greenhouse Data..."):
            results = model.predict(img_array, conf=confidence)
            annotated = results[0].plot()

        with col_left:
            st.subheader("AI Vision Feed")
            st.image(annotated, use_column_width=True, caption="Detected Tomatoes & Flowers")

        with col_right:
            st.subheader("📊 Yield Statistics")
            
            if results[0].boxes:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(classes)

                # Metric Cards
                for cls_id, count in counts.items():
                    st.metric(label=names[cls_id].upper(), value=count)

                # Simple Bar Chart
                df = pd.DataFrame({
                    "Stage": [names[c] for c in counts.keys()],
                    "Count": list(counts.values())
                })
                st.bar_chart(df.set_index("Stage"))
            else:
                st.warning("No tomatoes or flowers detected in this frame.")

else:
    st.info("👋 Welcome! Please upload a photo or video to begin the AI analysis.")

