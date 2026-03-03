import os
import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
import playsound

# ---------------- SETTINGS ----------------
DRONE_IP = "192.168.42.1"
AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "audio")
YOLO_MODEL_PATH = "yolov8n.pt"

COOLDOWN_SECONDS = 30
MAX_FLIGHT_TIME = 3  # seconds
last_trigger_time = -COOLDOWN_SECONDS

# ---------------- AUDIO ----------------
def play_random_audio():
    wav_files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".wav")]
    if not wav_files:
        print("[AUDIO] No .wav files found!")
        return
    selected = random.choice(wav_files)
    path = os.path.join(AUDIO_FOLDER, selected)
    print(f"[AUDIO] Playing {selected}")
    playsound.playsound(path)

# ---------------- DRONE CAMERA ----------------
class DroneCamera:
    def __init__(self, drone_ip):
        self.drone = olympe.Drone(drone_ip)
        self.frame = None

    def start(self):
        self.drone.connect()
        self.drone.streaming.set_callbacks(raw_cb=self.on_raw_frame)
        self.drone.streaming.start()

    def stop(self):
        self.drone.streaming.stop()
        self.drone.disconnect()

    def on_raw_frame(self, yuv_frame):
        info = yuv_frame.info()
        height = info["raw"]["frame"]["info"]["height"]
        width = info["raw"]["frame"]["info"]["width"]
        yuv = np.frombuffer(yuv_frame.as_ndarray(), dtype=np.uint8)
        yuv = yuv.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        self.frame = bgr
        yuv_frame.unref()

# ---------------- TAKEOFF SEQUENCE ----------------
def safe_takeoff(max_flight_time=MAX_FLIGHT_TIME):
    start_time = time.time()
    print(f"[DRONE] Connecting to {DRONE_IP}...")
    drone = olympe.Drone(DRONE_IP)
    drone.connect()

    print("[DRONE] Taking off...")
    assert drone(TakeOff()).wait().success()
    print("[DRONE] Takeoff successful.")

    # Stabilize briefly
    time.sleep(1)

    # Enforce max flight time
    elapsed = time.time() - start_time
    remaining = max_flight_time - elapsed
    if remaining > 0:
        time.sleep(remaining)

    print("[DRONE] Landing...")
    assert drone(Landing()).wait().success()
    print("[DRONE] Landed successfully.")
    drone.disconnect()
    print("[DRONE] Disconnected.")

# ---------------- MAIN ----------------
def main():
    global last_trigger_time

    model = YOLO(YOLO_MODEL_PATH)
    cam = DroneCamera(DRONE_IP)
    cam.start()
    print("[SYSTEM] Drone camera started. Waiting for human detection...")

    sequence_done = False
    try:
        while not sequence_done:
            frame = cam.frame
            if frame is None:
                continue

            # Resize for YOLO
            small = cv2.resize(frame, (320, 240))
            results = model(small, verbose=False)

            # Detect humans
            human_detected = False
            for r in results:
                for cls in r.boxes.cls:
                    if model.names[int(cls)] == "person":
                        human_detected = True
                        break
                if human_detected:
                    break

            # Trigger sequence
            if human_detected and (time.time() - last_trigger_time > COOLDOWN_SECONDS):
                print("[DETECTION] Human detected!")
                play_random_audio()
                safe_takeoff()
                last_trigger_time = time.time()
                sequence_done = True
                break

            # Show annotated frame
            annotated = results[0].plot()
            cv2.imshow("Drone Human Detection", cv2.resize(annotated, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Done.")

if __name__ == "__main__":
    main()
