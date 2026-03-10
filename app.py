import streamlit as st
import requests
import ast

USER = "seth-netizen"
REPO = "GEMINI-tomato_app"
# URLs for the text and the image
TEXT_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/status.txt"
IMG_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/detections.jpg"

st.title("🍅 Greenhouse Live AI Scout")

# 1. Display the Image first for your presentation
st.subheader("Live Camera Feed (AI Vision)")
try:
    # Use a random parameter to force the browser to refresh the image
    st.image(f"{IMG_URL}?v={st.runtime.scriptrunner.add_script_run_ctx}", 
             caption="Real-time detections from Raspberry Pi", use_column_width=True)
except:
    st.warning("Loading live image...")

# 2. Display the Stats
def get_data():
    try:
        response = requests.get(f"{TEXT_URL}?v={st.runtime.scriptrunner.add_script_run_ctx}")
        if response.status_code == 200:
            return ast.literal_eval(response.text)
    except:
        return None

data = get_data()
if data:
    st.subheader("Current Counts")
    cols = st.columns(len(data))
    for i, (label, count) in enumerate(data.items()):
        cols[i].metric(label.upper(), count)