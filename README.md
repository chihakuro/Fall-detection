# 🚨 Real-Time Patient Fall Detection & Alert System

## 📌 Overview
This repository contains a real-time computer vision pipeline designed for healthcare monitoring and elderly care. The system continuously processes video feeds to detect instances of a patient falling. Upon detecting a fall, it immediately triggers a push notification to caregivers' mobile devices, ensuring rapid response times.

## ⚙️ Core Architecture & Technologies

The system is built upon a lightweight, high-performance edge computing pipeline:
* **Pose Estimation:** Utilizes **Google MediaPipe Pose** to extract 33 3D spatiotemporal landmarks from the human body in real-time.
* **Action Classification:** A custom Deep Learning model built with **TensorFlow/Keras** processes the sequential sequence of these skeletal landmarks to classify the current action and identify sudden falls.
* **Real-Time Alerting:** Integrates the **Pushover API** via standard HTTP requests to instantly dispatch critical alerts ("Patient fell!") directly to registered mobile devices.
* **Video Processing:** Uses **OpenCV** for frame capture, dynamic bounding box rendering, and real-time FPS overlay.

## 🚀 Pipeline Workflow
1. **Frame Capture:** OpenCV reads the real-time webcam/video feed.
2. **Feature Extraction:** MediaPipe processes the frame and outputs human skeletal coordinates.
3. **Sequence Buffering:** The coordinates are buffered into a sliding window array (`slidebar`) to capture the temporal dynamics of the movement.
4. **Inference:** The Keras model evaluates the spatial-temporal sequence.
5. **Trigger:** If a "Fall" state is classified, an HTTP POST request is sent to the Pushover API, pushing an emergency notification to the designated caregiver.

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/fall-detection-system.git
cd fall-detection-system
```

**2. Install dependencies:**
```bash
pip install opencv-python mediapipe numpy tensorflow keras
```

**3. Configure Environment Variables (Security Best Practice):**
Do NOT hardcode your API keys in the script. Export your Pushover credentials as environment variables or use a `.env` file before running the system:
```bash
export PUSHOVER_TOKEN="your_api_token_here"
export PUSHOVER_USER="your_user_key_here"
```
*(Note: Ensure you update the `push()` function in `main.py` to read from `os.environ` instead of plain text strings).*

**4. Run the system:**
```bash
python main.py
```
