import os
import time
import random
import cv2
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO
import playsound
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

# ---------------- CONFIG ----------------
DRONE_IP = "192.168.42.1"
AUDIO_FOLDER = "./audio"
COOLDOWN_SECONDS = 30
HOVER_TIME = 1

# ---------------- STATE ----------------
current_frame = None
annotated_frame = None

is_flying = False
last_flight_time = -COOLDOWN_SECONDS
person_present = False
window_ready = False   # <-- ADDED

state_lock = Lock()

# ---------------- AUDIO ----------------
def play_random_audio():
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]
    if not files:
        return
    path = os.path.join(AUDIO_FOLDER, random.choice(files))
    playsound.playsound(path)

# ---------------- DRONE ----------------
drone = olympe.Drone(DRONE_IP)

def start_drone():
    drone.connect()
    drone.streaming.set_callbacks(raw_cb=on_raw_frame)
    drone.streaming.start()
    print("[DRONE] Connected & streaming")

def stop_drone():
    drone.streaming.stop()
    drone.disconnect()

# ---------------- VIDEO CALLBACK ----------------
def on_raw_frame(yuv_frame):
    global current_frame
    info = yuv_frame.info()
    h = info["raw"]["frame"]["info"]["height"]
    w = info["raw"]["frame"]["info"]["width"]

    yuv = np.frombuffer(yuv_frame.as_ndarray(), dtype=np.uint8)
    yuv = yuv.reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    current_frame = bgr
    yuv_frame.unref()

# ---------------- FLIGHT THREAD ----------------
def flight_sequence():
    global is_flying, last_flight_time

    with state_lock:
        if is_flying:
            return
        is_flying = True

    print("[FLIGHT] Triggered")
    play_random_audio()

    drone(TakeOff()).wait().success()
    time.sleep(HOVER_TIME)
    drone(Landing()).wait().success()

    last_flight_time = time.time()
    print("[FLIGHT] Completed")

    with state_lock:
        is_flying = False

# ---------------- DETECTION ----------------
model = YOLO("yolov8n.pt")

def detection_loop():
    global annotated_frame, person_present

    while True:
        frame = current_frame
        if frame is None:
            time.sleep(0.01)
            continue

        resized = cv2.resize(frame, (320, 240))
        results = model(resized, classes=[0], conf=0.35, verbose=False)

        annotated = results[0].plot()
        annotated_frame = cv2.resize(annotated, (640, 480))

        detected = any(
            model.names[int(cls)] == "person"
            for r in results
            for cls in r.boxes.cls
        )

        now = time.time()

        with state_lock:
            if detected:
                if (
                    window_ready  # <-- ADDED
                    and not person_present
                    and not is_flying
                    and now - last_flight_time > COOLDOWN_SECONDS
                ):
                    Thread(target=flight_sequence, daemon=True).start()
                person_present = True
            else:
                person_present = False

        time.sleep(0.03)

# ---------------- MAIN ----------------
def main():
    global window_ready  # <-- ADDED

    start_drone()
    Thread(target=detection_loop, daemon=True).start()

    print("[SYSTEM] Running — press Q to quit")

    try:
        while True:
            if annotated_frame is not None:
                cv2.imshow("Drone Detection", annotated_frame)
                window_ready = True  # <-- ADDED

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_drone()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
