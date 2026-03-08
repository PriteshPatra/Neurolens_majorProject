# Neurolens - Real-Time Stress Detection & Mitigation System

Neurolens is a full-stack, AI-powered web application that monitors your facial expressions in real-time, infers your stress levels using a custom YOLOv8 model, and automatically mitigates that stress by dynamically adjusting your hardware's screen brightness and issuing native desktop alerts when you need a break.

## Features

- **Live Emotion Tracking:** Utilizes a custom YOLOv8 model to detect 8 distinct emotions.
- **Strict Accuracy Logic:** Employs a >60% confidence threshold and a 30-frame (5-second) rolling temporal window to ensure the system reacts only to sustained, genuine emotions, eliminating single-frame "jitter."
- **Native OS Brightness Scaling:** Cross-platform controllers automatically map your stress level (0-100) to your screen's brightness. (Supports Windows WMI, Linux Wayland/X11, and macOS).
- **Native OS Notifications:** Triggers actual desktop notifications whenever your stress level breaches the maximum threshold.
- **Dashboard interface:** Built with Next.js 14, complete with real-time Chart.js graphs, animated stress gauges, and adjustable settings logic.

## Prerequisites

- **Docker & Docker Compose:** For running the isolated containerized environment.
- **Python 3.11+:** For running the native backend (required if you want the app to physically alter your hardware screen brightness and trigger OS notifications).
- **Webcam:** Required for emotion tracking.

## Configuration (via UI)

Navigate to the `Settings` page on the web app to configure:
- **Min / Max Brightness:** The bounds between which the screen will adapt.
- **Smoothing Factor:** How smoothly the brightness transitions.
- **Stress Threshold & Notification Cooldown:** When and how often you get alerted to take a break.

---

## How to Run: Option A (Native Hardware Control - Recommended)
Because Docker is an isolated sandbox, a Dockerized backend **cannot** physically dim your monitor or trigger desktop notifications. To enable these features, run the backend natively:

1. **Start the Frontend via Docker:**
   ```bash
   docker compose up -d frontend
   ```
2. **Start the Backend Natively:**
   **Linux / macOS:**
   ```bash
   chmod +x run_native_backend.sh
   ./run_native_backend.sh
   ```
   **Windows:**
   ```cmd
   run_native_backend.bat
   ```
3. Open [http://localhost:3000](http://localhost:3000)

---

## How to Run: Option B (Fully Dockerized)
If you only want to view the dashboard and do not need the app to physically alter your computer's monitor brightness or send native desktop popups:

1. Ensure the native backend is **not** running (port 8001 must be free).
2. Start both services via Docker:
   ```bash
   docker compose up -d
   ```
3. Open [http://localhost:3000](http://localhost:3000)
*(Note: Browser-level Web Notifications will still work as a fallback system).*

## Troubleshooting

### Address Already in Use (Port 8001)
If you see `Error starting userland proxy: listen tcp4 0.0.0.0:8001: bind: address already in use` when running `docker compose up -d`, this means you have the native `./run_native_backend.sh` running in the background.
**Fix:** Find and stop the native Python process using port 8001, or only start the frontend manually (`docker compose up -d frontend`).

### Model Weights Missing
Ensure that your YOLOv8 weights are located in `runs/detect/emotion_model_train/weights/best.pt` relative to the project root.
