import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
import tempfile
import os

# -------------------------------------------------
# Page Config
# -------------------------------------------------

st.set_page_config(
    page_title="Greenhouse Scout AI",
    page_icon="🍅",
    layout="wide"
)

# -------------------------------------------------
# Custom Styling (for presentations)
# -------------------------------------------------

st.markdown("""
<style>
.main { background-color: #0e1117; }
[data-testid="stMetricValue"] { color: #ff4b4b; font-size: 28px; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_tomato_model():

    if not os.path.exists("best.pt"):
        st.error("❌ Model file 'best.pt' not found. Please upload it to the repository.")
        st.stop()

    return YOLO("best.pt")

model = load_tomato_model()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("🛠️ AI Control Panel")

mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["📁 Upload Image", "🎥 Upload Video", "📸 Live Camera"]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.25
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Lower confidence helps detect small tomato flowers.")

# -------------------------------------------------
# Main Interface
# -------------------------------------------------

st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.markdown("### Real-time Yield Analysis & Ripeness Tracking")

col_left, col_right = st.columns([2, 1])

source_data = None

# -------------------------------------------------
# Input Selection
# -------------------------------------------------

if mode == "📸 Live Camera":
    source_data = st.camera_input("Capture Greenhouse Frame")

elif mode == "📁 Upload Image":
    source_data = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

elif mode == "🎥 Upload Video":
    source_data = st.file_uploader(
        "Upload Greenhouse Walkthrough Video",
        type=["mp4", "mov", "avi"]
    )

# -------------------------------------------------
# Processing
# -------------------------------------------------

if source_data:

    # -------------------------------
    # VIDEO PROCESSING
    # -------------------------------

    if mode == "🎥 Upload Video":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_data.read())

        video = cv2.VideoCapture(tfile.name)

        with col_left:
            st_frame = st.empty()

        with col_right:
            st.subheader("📊 Live Tracking")
            metric_placeholder = st.empty()

        while video.isOpened():

            ret, frame = video.read()

            if not ret:
                break

            results = model.track(
                frame,
                conf=confidence,
                persist=True,
                verbose=False
            )

            annotated_frame = results[0].plot()

            st_frame.image(
                annotated_frame,
                channels="BGR",
                use_column_width=True
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:

                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(classes)

                with metric_placeholder.container():

                    cols = st.columns(min(4, len(counts)))

                    for i, (cls_id, count) in enumerate(counts.items()):
                        cols[i % len(cols)].metric(
                            names[cls_id].upper(),
                            count
                        )

        video.release()

    # -------------------------------
    # IMAGE PROCESSING
    # -------------------------------

    else:

        image = Image.open(source_data)
        img_array = np.array(image)

        with st.spinner("🔍 Analyzing Greenhouse Data..."):

            results = model.predict(
                img_array,
                conf=confidence
            )

            annotated = results[0].plot()

        with col_left:

            st.subheader("AI Vision Feed")

            st.image(
                annotated,
                use_column_width=True,
                caption="Detected Tomatoes & Flowers"
            )

        with col_right:

            st.subheader("📊 Yield Statistics")

            if results[0].boxes is not None and len(results[0].boxes) > 0:

                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(classes)

                # Metric cards
                cols = st.columns(min(4, len(counts)))

                for i, (cls_id, count) in enumerate(counts.items()):
                    cols[i % len(cols)].metric(
                        names[cls_id].upper(),
                        count
                    )

                # Chart
                df = pd.DataFrame({
                    "Stage": [names[c] for c in counts.keys()],
                    "Count": list(counts.values())
                })

                st.bar_chart(df.set_index("Stage"))

            else:

                st.warning("No tomatoes or flowers detected in this frame.")

# -------------------------------------------------
# Start Message
# -------------------------------------------------

else:

    st.info("👋 Welcome! Upload a greenhouse image or video to begin AI analysis.")