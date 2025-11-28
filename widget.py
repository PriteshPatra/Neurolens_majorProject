import time
import cv2
import os
import collections
import threading
import pystray  # <-- THIS LINE WAS MISSING
from PIL import Image, ImageDraw

from ultralytics import YOLO
import screen_brightness_control as sbc

# --- This section is copied from your Streamlit app ---
# --- IMPORTANT: The paths here are now relative to this widget.py file ---
base_path = os.path.dirname(os.path.abspath(__file__))
# Note: You might need to adjust this path if your 'runs' folder is elsewhere
model_path = os.path.join(base_path, "runs", "detect", "emotion_model_train", "weights", "best.pt") 

STRESS_MAPPING = { "Happy": 0, "Neutral": 10, "Surprise": 60, "Sad": 75, "Contempt": 80, "Disgust": 85, "Fear": 95, "Anger": 100, "N/A": 0 }

# --- GLOBAL VARIABLES FOR THE WIDGET ---
is_running = True
# Default values for configuration (can be passed from Streamlit in a more advanced version)
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 100
SMOOTHING_FACTOR = 0.1
current_brightness = 100

# --- THE CORE LOGIC IN A SEPARATE THREAD ---
def monitoring_thread(icon):
    global is_running, current_brightness

    # Load the model
    model = YOLO(model_path)
    model.model.names = { 0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprise" }
    
    # Initialize prediction history for smoothing
    prediction_history = collections.deque(maxlen=15)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        is_running = False
        icon.stop()
        return

    while is_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Run prediction
        results = model.predict(source=frame, verbose=False, conf=0.25)
        
        emotion_label = "N/A"
        stress_level = 0
        
        if results[0].boxes:
            box = results[0].boxes[results[0].boxes.conf.argmax()]
            raw_emotion_label = model.model.names[int(box.cls[0])]
            
            prediction_history.append(raw_emotion_label)
            most_common_emotion = collections.Counter(prediction_history).most_common(1)[0][0]
            emotion_label = most_common_emotion
            stress_level = STRESS_MAPPING.get(emotion_label, 0)

            # Brightness control
            target_brightness = MAX_BRIGHTNESS - (stress_level / 100) * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            current_brightness = (1 - SMOOTHING_FACTOR) * current_brightness + SMOOTHING_FACTOR * target_brightness
            sbc.set_brightness(int(current_brightness))
        else:
            prediction_history.clear()

        # Update the hover text of the system tray icon
        icon.title = f"Emotion: {emotion_label}\nStress: {stress_level}%"
        
        time.sleep(0.5) # Process roughly 2 frames per second to save resources

    # Cleanup
    cap.release()
    print("Monitoring stopped.")
    icon.stop()

# --- SYSTEM TRAY ICON SETUP ---
def on_exit(icon, item):
    global is_running
    is_running = False
    print("Exit command received...")

def create_icon_image():
    # Create a simple placeholder image for the icon
    # You could replace this with a loaded image file: Image.open("icon.png")
    image = Image.new('RGB', (64, 64), 'black')
    dc = ImageDraw.Draw(image)
    dc.text((25, 20), "🧠", fill='white') # Brain emoji
    return image

if __name__ == "__main__":
    icon_image = create_icon_image()
    menu = (pystray.MenuItem('Exit', on_exit),)
    icon = pystray.Icon("StressDetector", icon_image, "Stress Detector", menu)

    # Run the monitoring in a separate thread
    monitor = threading.Thread(target=monitoring_thread, args=(icon,))
    monitor.start()

    # Run the icon
    icon.run()