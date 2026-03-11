import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load your 4th-year project model
model = YOLO('best.pt') 

st.title("🍅 Greenhouse Scout: Live AI Mode")

# Use Streamlit's built-in camera widget for better mobile compatibility
img_file = st.camera_input("Take a photo or scan tomatoes")

if img_file:
    # Convert the file to an image the AI can read
    img = Image.open(img_file)
    img_array = np.array(img)

    # Run YOLO detection
    results = model(img_array)
    
    # Draw the bounding boxes (Bloom, Green, Red, etc.)
    annotated_frame = results[0].plot()
    
    # Display the result
    st.image(annotated_frame, caption="AI Detection Results")
    
    # Show counts for your presentation
    counts = results[0].boxes.cls.tolist()
    names = results[0].names
    report = {names[int(c)]: counts.count(c) for c in set(counts)}
    
    st.subheader("Yield Analysis")
    cols = st.columns(len(report) if report else 1)
    if report:
        for i, (label, count) in enumerate(report.items()):
            cols[i].metric(label.upper(), count)
    else:
        st.write("No tomatoes detected in this frame.")