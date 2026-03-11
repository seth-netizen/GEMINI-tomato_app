import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import Counter
import tempfile

# ------------------ Page Config ------------------
st.set_page_config(page_title="Greenhouse Scout AI", page_icon="🍅", layout="wide")

# Dark Theme Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #ff4b4b; font-size: 28px; }
    .stAlert { border-radius: 10px; }
    .css-1r6slb0 { background-color: #161b22; border-radius: 10px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# ------------------ Model Loader ------------------
@st.cache_resource
def load_tomato_model():
    return YOLO("best.pt")

model = load_tomato_model()

# ------------------ Sidebar ------------------
st.sidebar.title("🛠️ AI Control Panel")
mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["🎥 Upload Video", "📁 Upload Image", "📸 Live Camera"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.30)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Presentation Tip:** Use 'Track' mode (Video) to show the panel how the AI remembers unique tomatoes as the camera moves.")

# ------------------ Main Interface ------------------
st.title("🍅 Greenhouse Scout: Real-Time AI Vision")
st.markdown("### Persistent Object Tracking & Yield Analysis")

col_left, col_right = st.columns([2, 1])
source_data = None

# ------------------ Input ------------------
if mode == "📸 Live Camera":
    source_data = st.camera_input("Capture Greenhouse Frame")
elif mode == "📁 Upload Image":
    source_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif mode == "🎥 Upload Video":
    source_data = st.file_uploader("Upload Greenhouse Walkthrough", type=["mp4", "mov", "avi"])

# ------------------ Processing ------------------
if source_data:
    if mode == "🎥 Upload Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_data.read())
        vid = cv2.VideoCapture(tfile.name)

        # Placeholders for the "Live Feed" effect
        st_frame = col_left.empty()
        metric_placeholder = col_right.container()
        
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            # USE TRACK INSTEAD OF PREDICT
            # persist=True ensures the AI tracks the same tomato across frames
            results = model.track(frame, conf=confidence, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Draw persistent boxes and IDs
            annotated_frame = results[0].plot() 

            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, use_column_width=True)

            # Live Statistics Update
            if results[0].boxes.id is not None:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(classes)

                with metric_placeholder:
                    st.subheader("📊 Live Tracking Stats")
                    m_cols = st.columns(2)
                    for i, (cls_id, count) in enumerate(counts.items()):
                        m_cols[i % 2].metric(names[cls_id].upper(), count)
                    
                    # Live Bar Chart
                    df = pd.DataFrame({
                        "Stage": [names[c] for c in counts.keys()],
                        "Count": list(counts.values())
                    })
                    st.bar_chart(df.set_index("Stage"), height=250)
            
        vid.release()

    else:
        # Image/Camera Logic (Same as your current version but cleaned up)
        image = Image.open(source_data)
        img_array = np.array(image)
        with st.spinner("🔍 Analyzing..."):
            results = model.predict(img_array, conf=confidence)
            annotated = results[0].plot()
            col_left.image(annotated, use_column_width=True)
            
            if results[0].boxes:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(classes)
                for cls_id, count in counts.items():
                    col_right.metric(names[cls_id].upper(), count)
else:
    st.info("👋 Ready for scouting. Please upload a video or photo.")
