import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Greenhouse Scout AI", layout="wide")

st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.write("Scan tomatoes in your greenhouse and get instant yield analysis.")

# -----------------------
# Load Model
# -----------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------
# Sidebar Controls
# -----------------------

st.sidebar.title("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.25
)

mode = st.sidebar.radio(
    "Select Image Source",
    ["Camera", "Upload Image"]
)

# -----------------------
# Image Input
# -----------------------

image = None

if mode == "Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image)

else:
    uploaded = st.file_uploader(
        "Upload greenhouse image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        image = Image.open(uploaded)

# -----------------------
# Detection
# -----------------------

if image:

    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    with st.spinner("Running AI detection..."):
        results = model(img_array, conf=confidence)

    annotated = results[0].plot()

    st.subheader("Detection Results")
    st.image(annotated, use_column_width=True)

    # -----------------------
    # Yield Analysis
    # -----------------------

    st.subheader("Yield Analysis")

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        counts = Counter(classes)

        cols = st.columns(len(counts))

        for i,(cls_id,count) in enumerate(counts.items()):
            cols[i].metric(names[cls_id].upper(), count)

        # Chart
        data = {
            "Class":[names[c] for c in counts.keys()],
            "Count":[counts[c] for c in counts.keys()]
        }

        df = pd.DataFrame(data)

        st.bar_chart(df.set_index("Class"))

    else:
        st.warning("No tomatoes detected.")