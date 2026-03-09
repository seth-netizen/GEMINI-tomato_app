import streamlit as st
import requests
import ast
import pandas as pd

# 1. Configuration - Replace with your GitHub username and repo name
USER = "seth-netizen"
REPO = "GEMINI-tomato_app"
FILE = "status.txt"
URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/{FILE}"

st.set_page_config(page_title="Tomato Yield Scout", page_icon="🍅")

st.title("🍅 Greenhouse Yield Dashboard")
st.write("Real-time monitoring from Meru University Greenhouse")

def fetch_data():
    try:
        # We add a random parameter to the URL to bypass GitHub's cache
        response = requests.get(f"{URL}?nocache={st.timestamp()}")
        if response.status_code == 200:
            return ast.literal_eval(response.text)
    except:
        return None
    return None

# 2. Display the Data
data = fetch_data()

if data:
    # Show high-level metrics
    cols = st.columns(3)
    cols[0].metric("Red (Ready)", data.get("red", 0))
    cols[1].metric("Turning", data.get("turning", 0))
    cols[2].metric("Damaged ⚠️", data.get("damaged tomatoes", 0))
    
    # Show a full breakdown table
    st.subheader("Full Crop Breakdown")
    df = pd.DataFrame(list(data.items()), columns=['Stage', 'Count'])
    st.bar_chart(df.set_index('Stage'))
else:
    st.warning("Waiting for data from Raspberry Pi...")

if st.button('🔄 Refresh Live Data'):
    st.rerun()