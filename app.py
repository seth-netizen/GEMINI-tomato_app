import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px  # Professional interactive charts

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Agri-Vision: Greenhouse Intelligence",
    layout="wide",
    page_icon="🍅"
)

# 2. ENHANCED CSS (Mobile Optimized)
st.markdown("""
<style>
    .main { background-color: #f0f4f0; }
    .stMetric { background: white; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .report-box { padding: 20px; border-radius: 15px; color: white; margin-bottom: 10px; }
    h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

ALL_CLASSES = ["Bloom Flower", "Wilted Flower", "Immature Green", "Mature Green", "Turning", "Light Red", "Red", "Damaged"]

# 4. HEADER
st.title("🍅 Greenhouse Intelligence System")
st.markdown("---")

# 5. INPUT SECTION
tab1, tab2 = st.tabs(["📁 Batch Upload", "📸 Live Camera"])
with tab1:
    uploaded_files = st.file_uploader("Upload greenhouse images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
with tab2:
    camera_photo = st.camera_input("Capture greenhouse image")

images_to_process = []
if uploaded_files: images_to_process = [Image.open(f) for f in uploaded_files]
elif camera_photo: images_to_process = [Image.open(camera_photo)]

# 6. DETECTION
if images_to_process:
    master_counts = {cls: 0 for cls in ALL_CLASSES}

    for idx, img in enumerate(images_to_process):
        results = model(img)
        plotted = results[0].plot()[:, :, ::-1]
        st.image(plotted, caption=f"Analysis View {idx+1}", use_column_width=True)

        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls)]
            master_counts[cls_name] = master_counts.get(cls_name, 0) + 1

    # 7. ANALYTICS
    st.markdown("## 📊 Yield Analytics")
    df = pd.DataFrame(list(master_counts.items()), columns=["Stage", "Count"])
    total_detected = df["Count"].sum()
    
    # KPIs (Key Performance Indicators) for the Professor
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Detected", total_detected)
    kpi2.metric("Harvest Ready (Red)", master_counts.get("Red", 0))
    kpi3.metric("Health Alerts", master_counts.get("Damaged", 0), delta_color="inverse")

    # 8. PIE & BAR CHART (Interactive)
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_pie = px.pie(df, values='Count', names='Stage', title="Ripeness Profile", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_chart2:
        st.bar_chart(df.set_index("Stage"))

    # 9. HARVEST ADVISORY
    st.markdown("---")
    st.header("🌾 Harvest Advisory Report")
    
    red_count = master_counts.get("Red", 0)
    harvest_ratio = (red_count / total_detected * 100) if total_detected > 0 else 0

    if harvest_ratio >= 60:
        st.success(f"**Action: Harvest Immediately.** {harvest_ratio:.1f}% crop maturity reached.")
    elif harvest_ratio >= 30:
        st.warning(f"**Action: Selective Harvesting.** {harvest_ratio:.1f}% maturity. Monitor daily.")
    else:
        st.info(f"**Action: Growth Phase.** {harvest_ratio:.1f}% maturity. Estimated 5-7 days to harvest.")

    if master_counts.get("Damaged", 0) > 0:
        st.error(f"🚨 **Urgent:** {master_counts['Damaged']} damaged tomatoes found. Check for pests or calcium deficiency.")