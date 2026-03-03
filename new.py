from ultralytics import YOLO
import cv2
import time
import random
import os
from playsound import playsound
from actual import test_takeoff  # Your drone sequence

# Paths
AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "audio")

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Flight control variables
cooldown_seconds = 45  # Longer cooldown to avoid repeated flights
last_trigger_time = 0
human_present = False  # Track if a human is already present

def play_random_audio():
    """Play one random audio file from the audio/ folder."""
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith((".mp3", ".wav"))]
    if not audio_files:
        print("No audio files found in audio/ folder.")
        return
    selected = random.choice(audio_files)
    path = os.path.join(AUDIO_FOLDER, selected)
    print(f"🎵 Playing {selected}...")
    playsound(path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster detection
    small_frame = cv2.resize(frame, (320, 240))
    results = model(small_frame, verbose=False)

    # Detect humans
    human_detected = False
    for r in results:
        for c in r.boxes.cls:
            if model.names[int(c)] == "person":
                human_detected = True
                break
        if human_detected:
            break

    current_time = time.time()

    # Trigger flight only if a new human appears and cooldown has passed
    if human_detected and not human_present and (current_time - last_trigger_time > cooldown_seconds):
        human_present = True
        last_trigger_time = current_time
        print("Human detected — running drone sequence...")

        try:
            play_random_audio()  # 🔊 Play random clip before flight
            test_takeoff()       # 🛫 Drone takeoff sequence
        except Exception as e:
            print(f"Flight error: {e}")

        print("Flight finished. Waiting for next detection...")

    # Reset human_present when no human is detected
    if not human_detected:
        human_present = False

    # Display camera feed
    annotated_frame = results[0].plot()
    cv2.imshow("Human Detection", cv2.resize(annotated_frame, (640, 480)))

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
