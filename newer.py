import os
import time
import json
import cv2
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO
import olympe

from olympe.messages.ardrone3.PilotingState import (
    FlyingStateChanged,
    AltitudeChanged,
    AttitudeChanged,
    SpeedChanged,
    PositionChanged,
)
from olympe.messages.common.CommonState import BatteryStateChanged


# ---------------- CONFIG ----------------
DRONE_IP = "192.168.42.1"

# YOLO
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.35
YOLO_W, YOLO_H = 320, 240
DISP_W, DISP_H = 640, 480

# Logging
TELEMETRY_LOG_PATH = "./telemetry_log.jsonl"
DETECTION_LOG_PATH = "./detection_log.jsonl"
LOG_EVERY_N_TELEMETRY = 3   # log every N telemetry polls (keeps file smaller)

# “Height-follow” dry-run suggestions (NO DRONE MOVEMENT)
DEADBAND_PX = 10
GAZ_GAIN_DIV = 120.0
GAZ_CLAMP = 0.30

# Dead-reckoning (integrate velocity -> delta position)
INTEGRATE_SPEED = True
RESET_DELTAS_ON_LANDED = True


# ---------------- STATE ----------------
state_lock = Lock()

current_frame = None
annotated_frame = None
window_ready = False

# Telemetry values (polled)
flying_state = "unknown"
altitude_m = 0.0
battery_pct = 0
roll = pitch = yaw = 0.0

# Speed/position (best effort)
vx = vy = vz = 0.0            # m/s (if available)
gps_lat = gps_lon = 0.0
gps_alt = 0.0

# Integrated deltas (meters) - DRIFTY indoors
dx = dy = dz = 0.0
_last_speed_ts = None

# Detection values (for display/log)
person_bbox_h_px = 0.0
person_center_y_px = -1.0
gaz_suggestion = 0.0
center_err_px = 0.0


# ---------------- LOGGING ----------------
def jsonl_append(path: str, payload: dict):
    payload["ts"] = time.time()
    with open(path, "a", buffering=1) as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


# ---------------- DRONE ----------------
drone = olympe.Drone(DRONE_IP)

def on_raw_frame(yuv_frame):
    """Matches your working code exactly (raw_cb + NV12)."""
    global current_frame

    info = yuv_frame.info()
    h = info["raw"]["frame"]["info"]["height"]
    w = info["raw"]["frame"]["info"]["width"]

    yuv = np.frombuffer(yuv_frame.as_ndarray(), dtype=np.uint8)
    yuv = yuv.reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    with state_lock:
        current_frame = bgr

    yuv_frame.unref()

def start_drone():
    drone.connect()
    drone.streaming.set_callbacks(raw_cb=on_raw_frame)
    drone.streaming.start()
    print("[DRONE] Connected & streaming")

def stop_drone():
    try:
        drone.streaming.stop()
    except Exception:
        pass
    try:
        drone.disconnect()
    except Exception:
        pass
    print("[DRONE] Disconnected")


# ---------------- TELEMETRY HELPERS ----------------
def _safe_float(d: dict, *keys, default=0.0):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def _safe_int(d: dict, *keys, default=0):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                pass
    return int(default)

def extract_speed_components(speed_state: dict):
    """
    Different firmwares expose slightly different keys.
    We try common patterns.
    """
    # Most common in ardrone3.PilotingState.SpeedChanged
    sx = _safe_float(speed_state, "speedX", "vx", "x", default=0.0)
    sy = _safe_float(speed_state, "speedY", "vy", "y", default=0.0)
    sz = _safe_float(speed_state, "speedZ", "vz", "z", default=0.0)
    return sx, sy, sz


# ---------------- TELEMETRY (POLLING) ----------------
def telemetry_loop():
    """
    Poll telemetry using drone.get_state(...)
    """
    global flying_state, altitude_m, battery_pct, roll, pitch, yaw
    global vx, vy, vz, gps_lat, gps_lon, gps_alt
    global dx, dy, dz, _last_speed_ts

    i = 0
    while True:
        try:
            fs = drone.get_state(FlyingStateChanged)
            alt = drone.get_state(AltitudeChanged)
            att = drone.get_state(AttitudeChanged)
            bat = drone.get_state(BatteryStateChanged)

            spd = drone.get_state(SpeedChanged)
            pos = drone.get_state(PositionChanged)

            now = time.time()

            with state_lock:
                flying_state = fs.get("state", flying_state)
                altitude_m = float(alt.get("altitude", altitude_m))

                # BatteryStateChanged is common.CommonState
                battery_pct = int(bat.get("percent", battery_pct))

                roll = float(att.get("roll", roll))
                pitch = float(att.get("pitch", pitch))
                yaw = float(att.get("yaw", yaw))

                # Speed (often works indoors)
                _vx, _vy, _vz = extract_speed_components(spd if isinstance(spd, dict) else {})
                vx, vy, vz = _vx, _vy, _vz

                # GPS-ish position (often useless indoors, but we show it anyway)
                if isinstance(pos, dict):
                    gps_lat = _safe_float(pos, "latitude", default=gps_lat)
                    gps_lon = _safe_float(pos, "longitude", default=gps_lon)
                    gps_alt = _safe_float(pos, "altitude", default=gps_alt)

                # Integrate speed -> delta position (rough, drifts)
                if INTEGRATE_SPEED:
                    if _last_speed_ts is None:
                        _last_speed_ts = now
                    dt = now - _last_speed_ts
                    _last_speed_ts = now

                    # sanity clamp dt in case of pauses
                    if 0.0 < dt < 0.5:
                        dx += vx * dt
                        dy += vy * dt
                        dz += vz * dt

                # Optional reset when landed (so it doesn't accumulate forever)
                if RESET_DELTAS_ON_LANDED and flying_state in ("landed", "emergency", "usertakeoff"):
                    dx = dy = dz = 0.0

            # log sometimes
            i += 1
            if i % LOG_EVERY_N_TELEMETRY == 0:
                jsonl_append(TELEMETRY_LOG_PATH, {
                    "type": "telemetry",
                    "flying_state": flying_state,
                    "altitude_m": altitude_m,
                    "battery_pct": battery_pct,
                    "roll": roll, "pitch": pitch, "yaw": yaw,
                    "vx": vx, "vy": vy, "vz": vz,
                    "dx": dx, "dy": dy, "dz": dz,
                    "gps_lat": gps_lat, "gps_lon": gps_lon, "gps_alt": gps_alt,
                })

        except Exception as e:
            # if drone disconnects or something odd happens
            if i % 30 == 0:
                print("[TELEM] Poll error:", repr(e))

        time.sleep(0.1)


# ---------------- “HEIGHT FOLLOW” DRY RUN MATH ----------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def compute_gaz_suggestion(person_cy_px: float, frame_h_px: int):
    """
    DRY RUN only:
    - If person is above center, suggest going UP (positive gaz)
    - If person is below center, suggest going DOWN (negative gaz)
    This does NOT move the drone.
    """
    target = frame_h_px / 2.0
    err = target - person_cy_px
    if abs(err) < DEADBAND_PX:
        return 0.0, err
    gaz = clamp(err / GAZ_GAIN_DIV, -GAZ_CLAMP, +GAZ_CLAMP)
    return gaz, err


# ---------------- DETECTION ----------------
model = YOLO(YOLO_MODEL)

def get_largest_person(results):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    best = None
    best_area = 0.0

    for i in range(len(r.boxes)):
        cls = int(r.boxes.cls[i])
        if model.names.get(cls, "") != "person":
            continue

        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        area = (x2 - x1) * (y2 - y1)
        conf = float(r.boxes.conf[i])

        if area > best_area:
            best_area = area
            best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf}

    return best

def detection_loop():
    global annotated_frame, person_bbox_h_px, person_center_y_px, gaz_suggestion, center_err_px

    while True:
        with state_lock:
            frame = None if current_frame is None else current_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        resized = cv2.resize(frame, (YOLO_W, YOLO_H))
        results = model(resized, classes=[0], conf=YOLO_CONF, verbose=False)

        # Draw YOLO boxes and resize for display
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (DISP_W, DISP_H))

        person = get_largest_person(results)

        if person is not None:
            h_px = float(person["y2"] - person["y1"])
            cy = float((person["y1"] + person["y2"]) / 2.0)
            gaz, err = compute_gaz_suggestion(cy, YOLO_H)

            with state_lock:
                person_bbox_h_px = h_px
                person_center_y_px = cy
                gaz_suggestion = gaz
                center_err_px = err

            jsonl_append(DETECTION_LOG_PATH, {
                "type": "detect",
                "person": person,
                "bbox_h_px": h_px,
                "center_y_px": cy,
                "gaz_suggestion": gaz,
                "center_err_px": err
            })
        else:
            with state_lock:
                person_bbox_h_px = 0.0
                person_center_y_px = -1.0
                gaz_suggestion = 0.0
                center_err_px = 0.0

            jsonl_append(DETECTION_LOG_PATH, {"type": "detect", "person": None})

        # Overlay telemetry + dry-run info on the annotated image
        with state_lock:
            fs = flying_state
            alt = altitude_m
            bat = battery_pct
            rr, pp, yy = roll, pitch, yaw

            _vx, _vy, _vz = vx, vy, vz
            _dx, _dy, _dz = dx, dy, dz
            _lat, _lon, _galt = gps_lat, gps_lon, gps_alt

            h_px = person_bbox_h_px
            cy = person_center_y_px
            gaz = gaz_suggestion
            err = center_err_px

        def put(line, y):
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        put(f"State: {fs}", 25)
        put(f"Altitude (baro-ish): {alt:.2f} m", 50)
        put(f"Battery: {bat}%", 75)
        put(f"Attitude: R {rr:.2f}  P {pp:.2f}  Y {yy:.2f}", 100)

        put("---- SPEED / INDOOR MOTION (best effort) ----", 140)
        put(f"v (m/s): x {_vx:+.2f}  y {_vy:+.2f}  z {_vz:+.2f}", 165)
        put(f"Δ (m):  x {_dx:+.2f}  y {_dy:+.2f}  z {_dz:+.2f}", 190)
        put(f"GPS pos: lat {_lat:.6f} lon {_lon:.6f} alt {_galt:.1f}", 215)

        put("---- YOLO (person) DRY RUN ----", 255)
        put(f"bbox height(px): {h_px:.1f}", 280)
        put(f"centerY(px @ {YOLO_H}): {cy:.1f}", 305)
        put(f"gaz suggestion: {gaz:+.2f}  err(px): {err:+.1f}", 330)

        with state_lock:
            annotated_frame = annotated

        time.sleep(0.03)


# ---------------- MAIN ----------------
def main():
    global window_ready

    start_drone()

    Thread(target=telemetry_loop, daemon=True).start()
    Thread(target=detection_loop, daemon=True).start()

    print("[SYSTEM] Running — press Q to quit")
    print("[SYSTEM] Writes logs to telemetry_log.jsonl and detection_log.jsonl")

    try:
        while True:
            with state_lock:
                ann = None if annotated_frame is None else annotated_frame.copy()

            if ann is not None:
                cv2.imshow("Anafi USA: YOLO + Telemetry Overlay", ann)
                window_ready = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_drone()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()