import streamlit as st
import requests
import ast

# Your GitHub details
USER = "seth-netizen"
REPO = "GEMINI-tomato_app"
URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/status.txt"

st.title("🍅 Greenhouse Scout Dashboard")

def get_data():
    try:
        # Fetch the status.txt file your Pi just pushed
        response = requests.get(f"{URL}?nocache={st.runtime.scriptrunner.get_script_run_ctx()}")
        if response.status_code == 200:
            return ast.literal_eval(response.text)
    except:
        return None

data = get_data()

if data:
    st.subheader("Current Tomato Counts")
    cols = st.columns(len(data))
    for i, (label, count) in enumerate(data.items()):
        cols[i].metric(label.upper(), count)
    
    st.success("Data synced live from Raspberry Pi")
else:
    st.warning("Waiting for the first scan from the greenhouse...")

if st.button('Update Now'):
    st.rerun()