import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import urllib.request
import os

st.set_page_config(page_title="Greenhouse Scout AI", layout="wide")

st.title("🍅 Greenhouse Scout: Tomato Detection AI")
st.write("Scan tomatoes in your greenhouse and get instant yield analysis.")

# -------------------------
# Model Download Section
# -------------------------

MODEL_URL = "https://drive.google.com/file/d/1hOszfUyWzqEnH484qE_OBi1R4Q5njDVh/view?usp=drive_link"
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading AI model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded successfully!")

# -------------------------
# Load Model (cached)
# -------------------------

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# -------------------------
# Sidebar Controls
# -------------------------

st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25
)

mode = st.sidebar.radio(
    "Select Image Source",
    ["Camera", "Upload Image"]
)

# -------------------------
# Image Input
# -------------------------

image = None

if mode == "Camera":
    camera_image = st.camera_input("Take a photo of tomatoes")
    if camera_image:
        image = Image.open(camera_image)

else:
    uploaded_file = st.file_uploader(
        "Upload a greenhouse image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

# -------------------------
# Detection Section
# -------------------------

if image:

    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    with st.spinner("Running AI detection..."):

        results = model(img_array, conf=confidence)

        annotated_frame = results[0].plot()

    st.subheader("Detection Results")
    st.image(annotated_frame, use_column_width=True)

    # -------------------------
    # Yield Analysis
    # -------------------------

    st.subheader("🍅 Yield Analysis")

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        counts = Counter(classes)

        # Display metrics
        cols = st.columns(len(counts))

        for i, (cls_id, count) in enumerate(counts.items()):
            cols[i].metric(names[cls_id].upper(), count)

        # -------------------------
        # Chart Visualization
        # -------------------------

        data = {
            "Class": [names[c] for c in counts.keys()],
            "Count": [counts[c] for c in counts.keys()]
        }

        df = pd.DataFrame(data)

        st.subheader("Detection Distribution")
        st.bar_chart(df.set_index("Class"))

        # -------------------------
        # Harvest Insight
        # -------------------------

        st.subheader("🌱 Harvest Insight")

        total = sum(counts.values())

        st.write(f"Total tomatoes detected: **{total}**")

        # Example assumption: class 3 = red tomatoes
        ripe = counts.get(3, 0)

        if ripe > 0:
            st.success(f"{ripe} tomatoes are ready for harvest.")
        else:
            st.warning("No ripe tomatoes detected.")

    else:

        st.warning("No tomatoes detected in this image.")