import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
import tempfile
import time

# ------------------ Page Config ------------------
st.set_page_config(page_title="Greenhouse Scout AI", page_icon="🍅", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ff4b4b; font-size: 28px; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ------------------ Model Loader ------------------
@st.cache_resource
def load_tomato_model():
    return YOLO("best.pt")  # make sure your best.pt is in the repo

model = load_tomato_model()

# ------------------ Sidebar ------------------
st.sidebar.title("🛠️ AI Control Panel")
mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["📁 Upload Image", "🎥 Upload Video", "📸 Live Camera"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.sidebar.info("💡 Lower confidence detects smaller blooms but may increase false positives.")

# ------------------ Main Interface ------------------
st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.markdown("### Real-time Yield Analysis & Ripeness Tracking")

col_left, col_right = st.columns([2, 1])
source_data = None

# ------------------ Input ------------------
if mode == "📸 Live Camera":
    source_data = st.camera_input("Capture Greenhouse Frame")
elif mode == "📁 Upload Image":
    source_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif mode == "🎥 Upload Video":
    source_data = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

# ------------------ Video / Image Processing ------------------
if source_data:
    # --- Video Processing ---
    if mode == "🎥 Upload Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_data.read())
        vid = cv2.VideoCapture(tfile.name)

        st_frame = col_left.empty()
        metric_placeholder = col_right.empty()
        chart_placeholder = col_right.empty()

        # Loop over frames
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            # YOLO detection per frame
            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()  # Draw bounding boxes

            # Convert to RGB for Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, use_column_width=True)

            # Update class counts
            if results[0].boxes:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                counts = Counter(classes)
                names = results[0].names

                # Metric cards
                with metric_placeholder.container():
                    m_cols = st.columns(2)
                    for i, (cls_id, count) in enumerate(list(counts.items())[:4]):
                        m_cols[i % 2].metric(names[cls_id].upper(), count)

                # Bar chart
                df = pd.DataFrame({
                    "Stage": [names[c] for c in counts.keys()],
                    "Count": list(counts.values())
                })
                chart_placeholder.bar_chart(df.set_index("Stage"))

            # Small delay to simulate continuous video
            time.sleep(0.03)  # ~30 FPS

        vid.release()

    # --- Image / Camera Processing ---
    else:
        image = Image.open(source_data)
        img_array = np.array(image)

        with st.spinner("🔍 Analyzing Greenhouse Data..."):
            results = model.predict(img_array, conf=confidence)
            annotated = results[0].plot()

        col_left.subheader("AI Vision Feed")
        col_left.image(annotated, use_column_width=True, caption="Detected Tomatoes & Flowers")

        col_right.subheader("📊 Yield Statistics")
        if results[0].boxes:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            counts = Counter(classes)
            names = results[0].names

            # Metrics
            for cls_id, count in counts.items():
                col_right.metric(label=names[cls_id].upper(), value=count)

            # Bar chart
            df = pd.DataFrame({
                "Stage": [names[c] for c in counts.keys()],
                "Count": list(counts.values())
            })
            col_right.bar_chart(df.set_index("Stage"))
        else:
            col_right.warning("No tomatoes or flowers detected in this frame.")

else:
    st.info("👋 Welcome! Please upload a photo or video to begin the AI analysis.")