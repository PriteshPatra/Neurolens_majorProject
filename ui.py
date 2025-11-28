import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import queue
import time
import cv2
import os
import tkinter as tk

from ultralytics import YOLO
import screen_brightness_control as sbc

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GET ABSOLUTE PATHS FOR ASSETS ---
base_path = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(base_path, "logo.png")

# --- THIS IS THE UPDATED MODEL PATH ---
model_path = os.path.join(base_path, "runs", "detect", "emotion_model_train", "weights", "best.pt")


# --- NOTIFICATION FUNCTION (using tkinter) ---
def show_warning_notification():
    """Creates a temporary, centered warning window that self-destructs."""
    window = tk.Tk()
    window.withdraw()
    popup = tk.Toplevel(window)
    popup.overrideredirect(True)
    popup.attributes('-topmost', True)
    label = tk.Label(popup, text="⚠️ High Stress Detected! ⚠️\nConsider taking a short break.",
                     font=("Arial", 18, "bold"), bg="red", fg="white", padx=20, pady=20)
    label.pack()
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
    y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
    popup.geometry(f'+{x}+{y}')
    popup.after(3000, window.destroy)
    window.mainloop()

# --- MODEL AND MAPPINGS ---
@st.cache_resource
def load_yolo_model():
    model = YOLO(model_path)
    model.model.names = { 0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprise" }
    return model

STRESS_MAPPING = { "Happy": 0, "Neutral": 10, "Surprise": 60, "Sad": 75, "Contempt": 80, "Disgust": 85, "Fear": 95, "Anger": 100 }
model = load_yolo_model()

# --- SIDEBAR CONTENT ---
st.sidebar.title("System Configuration")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
MIN_BRIGHTNESS = st.sidebar.slider("Minimum Brightness", 0, 100, 30)
MAX_BRIGHTNESS = st.sidebar.slider("Maximum Brightness", 0, 100, 100)
SMOOTHING_FACTOR = st.sidebar.slider("Brightness Smoothing Factor", 0.01, 1.0, 0.1)
STRESS_THRESHOLD = st.sidebar.slider("Stress Notification Threshold", 0, 100, 80)
NOTIFICATION_COOLDOWN = st.sidebar.slider("Notification Cooldown (seconds)", 5, 60, 10)

# --- SESSION STATE INITIALIZATION ---
if "current_brightness" not in st.session_state: st.session_state.current_brightness = 100
if "last_notification_time" not in st.session_state: st.session_state.last_notification_time = 0
if "stress_history_chart" not in st.session_state: st.session_state.stress_history_chart = []
if "full_stress_history_session" not in st.session_state: st.session_state.full_stress_history_session = []
results_queue = queue.Queue()

# --- WEBCAM PROCESSING CALLBACK ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    if "current_brightness" not in st.session_state: st.session_state.current_brightness = 100
    if "last_notification_time" not in st.session_state: st.session_state.last_notification_time = 0
    try:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        results = model.predict(source=img, verbose=False, conf=0.4)
        annotated_frame = results[0].plot()
        emotion_label, stress_level, confidence = "N/A", 0, 0
        if results[0].boxes:
            box = results[0].boxes[results[0].boxes.conf.argmax()]
            emotion_label = model.model.names[int(box.cls[0])]
            stress_level = STRESS_MAPPING.get(emotion_label, 10)
            confidence = float(box.conf[0])
            target_brightness = MAX_BRIGHTNESS - (stress_level / 100) * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            st.session_state.current_brightness = (1 - SMOOTHING_FACTOR) * st.session_state.current_brightness + SMOOTHING_FACTOR * target_brightness
            sbc.set_brightness(int(st.session_state.current_brightness))
            
            current_time = time.time()
            if stress_level >= STRESS_THRESHOLD and (current_time - st.session_state.last_notification_time > NOTIFICATION_COOLDOWN):
                show_warning_notification()
                st.session_state.last_notification_time = current_time
                
        results_queue.put({"emotion": emotion_label, "stress": stress_level, "confidence": confidence})
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    except Exception as e:
        print(f"Error in video callback: {e}")
        return frame

# --- MAIN APP INTERFACE ---
st.title("🧠 Real-Time Stress Detection & Mitigation System")
col1, col2 = st.columns([2, 1.2])

with col1:
    st.header("Webcam Feed")
    webrtc_streamer(key="webcam", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

with col2:
    st.header("Live Analysis")
    emotion_placeholder = st.empty()
    status_placeholder = st.empty()
    st.write("---")
    st.write("**Real-Time Data**")
    stress_placeholder = st.empty()
    chart_placeholder = st.empty()
    st.write("---")
    st.write("**Session Analytics**")
    avg_stress_placeholder = st.empty()
    peak_stress_placeholder = st.empty()
    
    while True:
        try:
            result = results_queue.get(timeout=1.0)
            stress = result["stress"]
            
            confidence_text = f"{result['confidence']:.2f} Confidence"
            emotion_placeholder.metric("Detected Emotion", result["emotion"], delta=confidence_text)
            
            if stress < 40:
                status_placeholder.success(f"✔️ Status: Calm ({stress}%)")
            elif 40 <= stress < 75:
                status_placeholder.warning(f"⚠️ Status: Moderate ({stress}%)")
            else:
                status_placeholder.error(f"🚨 Status: High Stress ({stress}%)")
            
            stress_placeholder.progress(stress)
            
            st.session_state.stress_history_chart.append(stress)
            if len(st.session_state.stress_history_chart) > 100:
                st.session_state.stress_history_chart.pop(0)
            chart_placeholder.line_chart(st.session_state.stress_history_chart)
            
            st.session_state.full_stress_history_session.append(stress)
            avg_stress = np.mean(st.session_state.full_stress_history_session)
            peak_stress = np.max(st.session_state.full_stress_history_session)
            avg_stress_placeholder.metric("Average Session Stress", f"{avg_stress:.1f}%")
            peak_stress_placeholder.metric("Peak Session Stress", f"{peak_stress}%")

        except queue.Empty:
            time.sleep(0.1)