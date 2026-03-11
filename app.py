import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
import tempfile
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Greenhouse Scout AI", page_icon="🍅", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fff; }
    [data-testid="stMetricValue"] { color: #ff4b4b; font-size: 28px; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ------------------ MODEL LOADER ------------------
@st.cache_resource
def load_tomato_model():
    return YOLO("best.pt")  # Make sure best.pt is in the same folder

model = load_tomato_model()

# ------------------ SIDEBAR ------------------
st.sidebar.title("🛠️ AI Control Panel")
mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["📁 Upload Image", "🎥 Upload Video", "📸 Live Camera"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.info("💡 Use lower confidence for small blooms or hidden tomatoes.")

# ------------------ MAIN LAYOUT ------------------
st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.markdown("### Real-time Yield Analysis & Ripeness Tracking")
col_left, col_right = st.columns([2, 1])

source_data = None
if mode == "📸 Live Camera":
    source_data = st.camera_input("Capture Greenhouse Frame")
elif mode == "📁 Upload Image":
    source_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif mode == "🎥 Upload Video":
    source_data = st.file_uploader("Upload Greenhouse Walkthrough Video", type=["mp4", "mov", "avi"])

# ------------------ PROCESSING ------------------
if source_data:

    # -------- VIDEO PROCESSING --------
    if mode == "🎥 Upload Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_data.read())
        video = cv2.VideoCapture(tfile.name)

        st_frame = col_left.empty()
        stats_placeholder = col_right.empty()

        names = model.names
        total_counts = Counter()
        prev_time = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Run detection per frame
            results = model.predict(frame, conf=confidence, verbose=False)
            annotated_frame = results[0].plot()

            # Display frame
            st_frame.image(annotated_frame, channels="BGR", use_column_width=True)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Count detections
            frame_counts = Counter()
            if results[0].boxes:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                frame_counts = Counter(classes)
                for cls_id in frame_counts:
                    total_counts[cls_id] += frame_counts[cls_id]

            # Display live metrics
            with stats_placeholder.container():
                st.subheader(f"📊 Live Detection Counts (FPS: {int(fps)})")
                cols = st.columns(2)
                for i, (cls_id, count) in enumerate(total_counts.items()):
                    cols[i % 2].metric(names[cls_id].upper(), count)

        video.release()

    # -------- IMAGE OR CAMERA --------
    else:
        image = Image.open(source_data)
        img_array = np.array(image)

        with st.spinner("🔍 Analyzing Greenhouse Frame..."):
            results = model.predict(img_array, conf=confidence)
            annotated = results[0].plot()

        col_left.subheader("AI Vision Feed")
        col_left.image(annotated, use_column_width=True, caption="Detected Tomatoes & Flowers")

        col_right.subheader("📊 Yield Statistics")
        if results[0].boxes:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            counts = Counter(classes)
            names = results[0].names

            # Metric Cards
            for cls_id, count in counts.items():
                col_right.metric(names[cls_id].upper(), count)

            # Bar chart
            df = pd.DataFrame({"Stage": [names[c] for c in counts.keys()], "Count": list(counts.values())})
            col_right.bar_chart(df.set_index("Stage"))
        else:
            col_right.warning("No tomatoes or flowers detected in this frame.")

else:
    st.info("👋 Welcome! Please upload a photo or video to begin AI analysis.")