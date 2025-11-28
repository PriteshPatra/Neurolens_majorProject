import streamlit as st
import os
import subprocess
import sys

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- GET ABSOLUTE PATH FOR THE WIDGET SCRIPT ---
base_path = os.path.dirname(os.path.abspath(__file__))
widget_path = os.path.join(base_path, "widget.py")

# --- SIDEBAR AND MAIN INTERFACE ---
st.sidebar.title("System Configuration")
st.sidebar.info("Configuration for the monitoring widget is set within the `widget.py` script. This interface is now used only to start the background process.")

st.title("🧠 Stress Detection & Mitigation System")
st.write("Click the button below to start the monitoring service. An icon will appear in your system tray (usually at the bottom-right of your screen).")
st.write("You can then close this browser window.")

if st.button("🚀 Start Monitoring Widget"):
    if os.path.exists(widget_path):
        # Use subprocess.Popen to run the script in the background
        subprocess.Popen([sys.executable, widget_path])
        st.success("Monitoring widget started! Look for its icon in your system tray.")
        st.info("To stop the service, right-click the icon in the tray and select 'Exit'.")
    else:
        st.error(f"Error: `widget.py` not found at {widget_path}")