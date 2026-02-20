import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Tomato Ripeness AI", page_icon="🍅")

# 1. Load your YOLO model
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

st.title("🍅 Tomato Ripeness Detector")
st.write("Upload a photo to detect and classify tomatoes.")

# 2. Image Input
img_file = st.file_uploader("Upload or Capture Image", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Everything inside this block must be indented
    image = Image.open(img_file)
    st.image(image, caption='Target Image', use_column_width=True)
    
    if st.button("Run Detection"):
        # Everything inside the button must be indented further
        results = model(image)
        
        # Draw boxes on the image
        annotated_img = results[0].plot() 
        
        # Convert BGR to RGB so colors look natural
        annotated_img_rgb = annotated_img[:, :, ::-1] 
        st.image(annotated_img_rgb, caption="Detected Results", use_column_width=True)
        
        st.subheader("Detection Summary:")
        
        # Loop through detected boxes
        for box in results[0].boxes:
            confidence = box.conf.item()
            label_index = int(box.cls.item())
            label_name = results[0].names[label_index]
            
            st.write(f"✅ Found: **{label_name}** ({confidence:.2%})")

            # Your 60% color rule for the Red class
            if label_name.lower() == "red" and confidence < 0.60:
                st.warning(f"Note: This {label_name} tomato is below the 60% ripeness threshold.")