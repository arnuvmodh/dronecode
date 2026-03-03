import os
import time
import random
import cv2
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO
import playsound
import olympe
import tkinter as tk

from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import (
    SpeedChanged,
    FlyingStateChanged,
)

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
state_lock = Lock()

# Relative positions (meters)
rel_x = 0.0
rel_y = 0.0
rel_z = 0.0

# Last timestamp for speed integration
last_speed_time = None
speed_x = speed_y = speed_z = 0.0

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
    drone.subscribe(FlyingStateChanged())
    drone.subscribe(SpeedChanged())
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

# ---------------- TELEMETRY WINDOW ----------------
class TelemetryWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Drone Telemetry")
        self.label = tk.Label(
            self.root,
            text="Starting telemetry...",
            font=("Courier", 14),
            justify="left"
        )
        self.label.pack(padx=20, pady=20)
        self.update_telemetry()

    def update_telemetry(self):
        global rel_x, rel_y, rel_z, person_present
        try:
            flying_state = drone.get_state(FlyingStateChanged)
            text = (
                f"Flying State: {flying_state}\n"
                f"Relative Position (X,Y,Z): {rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f} m\n"
                f"{'Person detected!' if person_present else 'No person detected'}"
            )
        except Exception:
            text = f"Waiting for telemetry...\n{'Person detected!' if person_present else 'No person detected'}"

        self.label.config(text=text)
        self.root.after(100, self.update_telemetry)

# ---------------- FLIGHT THREAD ----------------
def flight_sequence():
    global is_flying, last_flight_time
    with state_lock:
        if is_flying:
            return
        is_flying = True

    print("[FLIGHT] Triggered")
    play_random_audio()

    # Flight sequence unchanged
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
            if detected and not is_flying and now - last_flight_time > COOLDOWN_SECONDS:
                Thread(target=flight_sequence, daemon=True).start()
            person_present = detected
        time.sleep(0.03)

# ---------------- SPEED CALLBACK ----------------
def speed_cb(msg):
    global speed_x, speed_y, speed_z, last_speed_time, rel_x, rel_y, rel_z
    now = time.time()
    dt = 0
    if last_speed_time is not None:
        dt = now - last_speed_time
    last_speed_time = now

    speed_x = msg.args["x"]
    speed_y = msg.args["y"]
    speed_z = msg.args["z"]

    # Integrate to get relative position
    if dt > 0:
        rel_x += speed_x * dt
        rel_y += speed_y * dt
        rel_z += speed_z * dt

# Subscribe speed callback
drone.on(SpeedChanged, speed_cb)

# ---------------- MAIN ----------------
def main():
    start_drone()
    gui = TelemetryWindow()
    Thread(target=detection_loop, daemon=True).start()
    print("[SYSTEM] Running — press Q to quit")

    def opencv_loop():
        if annotated_frame is not None:
            cv2.imshow("Drone Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_drone()
            cv2.destroyAllWindows()
            gui.root.quit()
            return
        gui.root.after(10, opencv_loop)

    opencv_loop()
    gui.root.mainloop()

if __name__ == "__main__":
    main()