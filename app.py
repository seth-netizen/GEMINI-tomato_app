import streamlit as st
import requests
import ast

# Configuration
USER = "seth-netizen"
REPO = "GEMINI-tomato_app"
FILE = "status.txt"
# We use the 'raw' URL to get the actual text
URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/{FILE}"

st.set_page_config(page_title="Tomato Scout", page_icon="🍅")
st.title("🍅 Greenhouse Yield Dashboard")

def fetch_data():
    try:
        # Adding a cache buster to get fresh data every time
        response = requests.get(f"{URL}?v={st.runtime.scriptrunner.add_script_run_ctx}")
        if response.status_code == 200:
            # Safely convert string to dictionary
            return ast.literal_eval(response.text)
    except Exception as e:
        st.error(f"Waiting for Pi data... Error: {e}")
        return None
    return None

data = fetch_data()

if data:
    st.subheader("Live Stats from Meru University")
    
    # Create metrics for your classes
    col1, col2, col3 = st.columns(3)
    col1.metric("Red (Ready)", data.get('red', 0))
    col2.metric("Green", data.get('green', 0))
    col3.metric("Damaged ⚠️", data.get('damaged tomatoes', 0))
    
    # Show the full raw report
    with st.expander("See full report"):
        st.write(data)
else:
    st.info("The Pi hasn't sent the first report yet. Please run scout.py on your Pi!")

if st.button('🔄 Refresh'):
    st.rerun()