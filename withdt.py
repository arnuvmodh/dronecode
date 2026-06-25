import os
import csv
import time
import random
import traceback
from threading import Thread, Lock, Event

import cv2
import numpy as np
from ultralytics import YOLO
import playsound
import olympe

from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from olympe.messages.ardrone3.PilotingState import (
    FlyingStateChanged,
    AltitudeChanged,
    SpeedChanged,
)
from olympe.messages.common.CommonState import BatteryStateChanged


# ================= CONFIG =================
DRONE_IP = "192.168.42.1"
AUDIO_FOLDER = "./audio"
MODEL_PATH = "yolov8n.pt"

COOLDOWN_SECONDS = 20
HOVER_TIME = 0.5
MAX_AIR_TIME = 2.0
TAKEOFF_TIMEOUT = 12.0
LAND_TIMEOUT = 12.0

TELEMETRY_CSV = "./flight_telemetry.csv"
TELEMETRY_HZ = 10.0
DETECTION_CONF = 0.35

FRAME_STALE_SECONDS = 0.75
STARTUP_ARM_DELAY = 0.5
WATCHDOG_PERIOD = 0.1

# Fast response for already-present people
PERSON_HOLD_SECONDS = 0.20
PERSON_LOST_GRACE_SECONDS = 0.40
DETECTION_INTERVAL = 0.05


# ================= SHARED STATE =================
current_frame = None
display_frame = None
last_frame_time = 0.0
window_ready = False
system_armed_time = 0.0

state_lock = Lock()
shutdown_event = Event()

is_flying = False
person_present = False
person_seen_since = None
last_person_detected_time = 0.0
last_flight_time = -COOLDOWN_SECONDS
flight_disabled = False
flight_start_time = None


# ================= DRONE =================
drone = olympe.AnafiUSA(DRONE_IP)


# ================= HELPERS =================
def safe_float(d, key, default=0.0):
    try:
        v = d.get(key, default)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def safe_int(d, key, default=0):
    try:
        v = d.get(key, default)
        return int(v) if v is not None else int(default)
    except Exception:
        return int(default)


def extract_speed(spd: dict):
    vx = vy = vz = 0.0
    if isinstance(spd, dict):
        for k in ("speedX", "vx", "x"):
            if k in spd:
                vx = safe_float(spd, k, 0.0)
                break
        for k in ("speedY", "vy", "y"):
            if k in spd:
                vy = safe_float(spd, k, 0.0)
                break
        for k in ("speedZ", "vz", "z"):
            if k in spd:
                vz = safe_float(spd, k, 0.0)
                break
    return vx, vy, vz


def state_str(v):
    try:
        return str(v)
    except Exception:
        return "unknown"


def disable_future_flights(reason):
    global flight_disabled
    with state_lock:
        flight_disabled = True
    print(f"[LOCKOUT] Future flights disabled: {reason}")


def get_flying_state_str():
    try:
        fs = drone.get_state(FlyingStateChanged) or {}
        return str(fs.get("state", "unknown"))
    except Exception:
        return "unknown"


def is_airborne_state():
    state = get_flying_state_str()
    return any(s in state for s in ("motor_ramping", "takingoff", "hovering", "flying", "landing"))


def is_landed_state():
    return get_flying_state_str() == "landed"


def frame_is_fresh():
    return (time.time() - last_frame_time) <= FRAME_STALE_SECONDS


def system_is_armed():
    return (time.time() - system_armed_time) >= STARTUP_ARM_DELAY


def cooldown_remaining():
    remaining = COOLDOWN_SECONDS - (time.time() - last_flight_time)
    return max(0.0, remaining)


def can_trigger_now():
    now = time.time()
    with state_lock:
        person_held_long_enough = (
            person_seen_since is not None
            and (now - person_seen_since) >= PERSON_HOLD_SECONDS
        )

        cooldown_elapsed = (now - last_flight_time) > COOLDOWN_SECONDS

        return (
            window_ready
            and system_is_armed()
            and frame_is_fresh()
            and person_present
            and person_held_long_enough
            and not is_flying
            and not flight_disabled
            and cooldown_elapsed
        )


def draw_status_overlay(frame):
    try:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 170), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        fresh = frame_is_fresh()
        armed = system_is_armed()
        airborne = is_airborne_state()
        ready = can_trigger_now()
        cooldown_left = cooldown_remaining()

        with state_lock:
            local_person_present = person_present
            local_is_flying = is_flying
            local_disabled = flight_disabled

        lines = [
            f"VIDEO: {'OK' if current_frame is not None else 'WAITING'}",
            f"FRESH: {'YES' if fresh else 'NO'}",
            f"ARMED: {'YES' if armed else 'NO'}",
            f"PERSON: {'YES' if local_person_present else 'NO'}",
            f"FLYING: {'YES' if local_is_flying or airborne else 'NO'}",
            f"COOLDOWN: {cooldown_left:.1f}s",
            f"READY: {'YES' if ready else 'NO'}",
            f"LOCKED: {'YES' if local_disabled else 'NO'}",
        ]

        y = 35
        for line in lines:
            color = (0, 255, 0)
            if "NO" in line or "WAITING" in line:
                color = (0, 200, 255)
            if line.startswith("READY: YES"):
                color = (0, 255, 0)
            if line.startswith("LOCKED: YES"):
                color = (0, 0, 255)

            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 18

        return frame

    except Exception as e:
        print(f"[OVERLAY] Error: {e}")
        return frame


def play_random_audio():
    try:
        files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".wav")]
        if not files:
            print("[AUDIO] No .wav files found")
            return
        path = os.path.join(AUDIO_FOLDER, random.choice(files))
        print(f"[AUDIO] Playing: {os.path.basename(path)}")
        playsound.playsound(path)
    except Exception as e:
        print(f"[AUDIO] Error: {e}")


def try_land(reason="unspecified", retries=3, wait_between=0.75):
    print(f"[SAFETY] Landing attempt started. Reason: {reason}")

    for attempt in range(1, retries + 1):
        if is_landed_state():
            print("[SAFETY] Already landed.")
            return True

        try:
            action = drone(Landing()).wait()
            ok = action.success()
            print(f"[SAFETY] Landing command attempt {attempt}: success={ok}")

            landed = drone(
                FlyingStateChanged(state="landed", _timeout=LAND_TIMEOUT)
            ).wait().success()

            if landed:
                print("[SAFETY] Confirmed landed.")
                return True

        except Exception as e:
            print(f"[SAFETY] Landing exception on attempt {attempt}: {e}")

        time.sleep(wait_between)

    print("[SAFETY] Landing could not be confirmed.")
    return False


# ================= STREAMING CALLBACK =================
def on_raw_frame(yuv_frame):
    global current_frame, display_frame, last_frame_time

    try:
        info = yuv_frame.info()
        h = info["raw"]["frame"]["info"]["height"]
        w = info["raw"]["frame"]["info"]["width"]

        yuv = np.array(yuv_frame.as_ndarray(), copy=True)
        yuv = yuv.reshape((h * 3 // 2, w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

        current_frame = bgr

        base = cv2.resize(bgr, (640, 480))
        if display_frame is None:
            display_frame = draw_status_overlay(base)

        last_frame_time = time.time()

    except Exception as e:
        print(f"[VIDEO] Frame callback error: {e}")


# ================= START / STOP =================
def start_drone():
    global system_armed_time

    print(f"[DRONE] Connecting to {DRONE_IP} ...")
    drone.connect()

    try:
        drone.streaming.set_callbacks(raw_cb=on_raw_frame)
        drone.streaming.start()
    except Exception:
        try:
            drone.disconnect()
        except Exception:
            pass
        raise

    system_armed_time = time.time()
    print("[DRONE] Connected and streaming")


def stop_drone():
    try:
        if is_airborne_state():
            try_land(reason="shutdown path", retries=4, wait_between=0.75)
    finally:
        try:
            drone.streaming.stop()
        except Exception as e:
            print(f"[DRONE] streaming.stop() warning: {e}")

        try:
            drone.disconnect()
        except Exception as e:
            print(f"[DRONE] disconnect() warning: {e}")

        print("[DRONE] Disconnected")


# ================= TELEMETRY =================
def telemetry_loop():
    header = [
        " ",
        "ts",
        "flying_state",
        "altitude_cm",
        "battery_pct",
        "vz_cm_s",
        "is_flying_flag",
        "person_present_flag",
        "flight_disabled_flag",
    ]

    period = 1.0 / TELEMETRY_HZ
    temp_path = TELEMETRY_CSV + ".tmp"

    while not shutdown_event.is_set():
        now = time.time()

        try:
            fs = drone.get_state(FlyingStateChanged) or {}
            alt = drone.get_state(AltitudeChanged) or {}
            bat = drone.get_state(BatteryStateChanged) or {}
            spd = drone.get_state(SpeedChanged) or {}

            flying_state = state_str(fs.get("state", "unknown")) if isinstance(fs, dict) else "unknown"

            altitude_m = safe_float(alt if isinstance(alt, dict) else {}, "altitude", 0.0)
            altitude_cm = altitude_m * 100.0

            battery_pct = safe_int(bat if isinstance(bat, dict) else {}, "percent", 0)

            _, _, vz = extract_speed(spd if isinstance(spd, dict) else {})
            vz_cm_s = vz * 100.0

            with state_lock:
                row = [
                    "1",
                    now,
                    flying_state,
                    altitude_cm,
                    battery_pct,
                    vz_cm_s,
                    int(is_flying),
                    int(person_present),
                    int(flight_disabled),
                ]

            with open(temp_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, TELEMETRY_CSV)

        except Exception as e:
            print(f"[TELEMETRY] Error: {e}")

        time.sleep(period)


# ================= WATCHDOG =================
def watchdog_loop():
    while not shutdown_event.is_set():
        try:
            with state_lock:
                local_is_flying = is_flying
                local_flight_start = flight_start_time
                local_disabled = flight_disabled

            if local_disabled:
                time.sleep(WATCHDOG_PERIOD)
                continue

            if is_airborne_state():
                if shutdown_event.is_set():
                    try_land(reason="watchdog: shutdown while airborne", retries=4)
                    disable_future_flights("shutdown while airborne")
                elif local_is_flying and local_flight_start is not None:
                    elapsed = time.time() - local_flight_start
                    if elapsed > MAX_AIR_TIME:
                        print(f"[WATCHDOG] Max air time exceeded: {elapsed:.2f}s")
                        landed_ok = try_land(reason="watchdog: max air time exceeded", retries=4)
                        if not landed_ok:
                            disable_future_flights("watchdog could not confirm landing")

            time.sleep(WATCHDOG_PERIOD)

        except Exception as e:
            print(f"[WATCHDOG] Error: {e}")
            time.sleep(WATCHDOG_PERIOD)


# ================= FLIGHT =================
def flight_sequence():
    global is_flying, last_flight_time, flight_start_time

    with state_lock:
        if is_flying or flight_disabled:
            return
        is_flying = True

    airborne = False
    print("[FLIGHT] Triggered")

    try:
        if not frame_is_fresh():
            raise RuntimeError("Refusing flight: video frame is stale")

        play_random_audio()

        if shutdown_event.is_set():
            raise RuntimeError("Shutdown requested before takeoff")

        action = drone(
            TakeOff()
            >> (
                FlyingStateChanged(state="hovering", _timeout=TAKEOFF_TIMEOUT)
                | FlyingStateChanged(state="flying", _timeout=TAKEOFF_TIMEOUT)
            )
        ).wait()

        if not action.success():
            try:
                detail = action.explain()
            except Exception:
                detail = "no explanation available"
            raise RuntimeError(f"Takeoff/hover expectation failed: {detail}")

        airborne = True
        with state_lock:
            flight_start_time = time.time()

        print("[FLIGHT] Airborne")

        while True:
            elapsed = time.time() - flight_start_time

            if elapsed >= HOVER_TIME:
                break

            if shutdown_event.is_set():
                raise RuntimeError("Shutdown requested during flight")

            time.sleep(0.05)

        landed_ok = try_land(reason="normal end of short flight", retries=3, wait_between=0.75)
        if not landed_ok:
            raise RuntimeError("Landing could not be confirmed")

        airborne = False
        with state_lock:
            last_flight_time = time.time()
        print("[FLIGHT] Completed successfully")

    except Exception as e:
        print(f"[FLIGHT] ERROR: {e}")
        traceback.print_exc()

        if airborne or is_airborne_state():
            landed_ok = try_land(reason=f"flight exception: {e}", retries=4, wait_between=0.75)
            if not landed_ok:
                disable_future_flights("abnormal flight; landing not confirmed")
        else:
            disable_future_flights("flight error before safe completion")

    finally:
        if is_airborne_state():
            landed_ok = try_land(reason="final safeguard", retries=4, wait_between=0.75)
            if not landed_ok:
                disable_future_flights("final safeguard failed")

        with state_lock:
            flight_start_time = None
            is_flying = False


# ================= DETECTION =================
model = YOLO(MODEL_PATH)

def detection_loop():
    global display_frame, person_present, person_seen_since, last_person_detected_time

    last_detection_time = 0.0

    while not shutdown_event.is_set():
        now = time.time()

        if now - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.005)
            continue

        frame = current_frame
        if frame is None:
            time.sleep(0.01)
            continue

        last_detection_time = now

        try:
            resized = cv2.resize(frame, (320, 240))
            results = model(resized, classes=[0], conf=DETECTION_CONF, verbose=False)

            annotated = cv2.resize(resized.copy(), (640, 480))
            detected_now = False

            scale_x = 640.0 / 320.0
            scale_y = 480.0 / 240.0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

                    for i, box in enumerate(xyxy):
                        cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
                        conf = float(confs[i]) if i < len(confs) else 0.0

                        if cls_id == 0:
                            detected_now = True
                            x1, y1, x2, y2 = box
                            X1 = int(x1 * scale_x)
                            Y1 = int(y1 * scale_y)
                            X2 = int(x2 * scale_x)
                            Y2 = int(y2 * scale_y)

                            cv2.rectangle(annotated, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                            label = f"person {conf:.2f}"
                            cv2.putText(
                                annotated,
                                label,
                                (X1, max(20, Y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

            start_flight = False

            with state_lock:
                if detected_now:
                    last_person_detected_time = now
                    if person_seen_since is None:
                        person_seen_since = now
                else:
                    if (now - last_person_detected_time) > PERSON_LOST_GRACE_SECONDS:
                        person_seen_since = None

                person_present = (now - last_person_detected_time) <= PERSON_LOST_GRACE_SECONDS

                person_held_long_enough = (
                    person_seen_since is not None
                    and (now - person_seen_since) >= PERSON_HOLD_SECONDS
                )

                can_trigger = (
                    window_ready
                    and system_is_armed()
                    and frame_is_fresh()
                    and person_present
                    and person_held_long_enough
                    and not is_flying
                    and not flight_disabled
                    and now - last_flight_time > COOLDOWN_SECONDS
                )

                if can_trigger:
                    start_flight = True
                    person_seen_since = now

            display_frame = draw_status_overlay(annotated)

            if start_flight:
                Thread(target=flight_sequence, daemon=False).start()

        except Exception as e:
            print(f"[DETECTION] Error: {e}")


# ================= MAIN =================
def main():
    global window_ready

    start_drone()

    telemetry_thread = Thread(target=telemetry_loop, daemon=True)
    watchdog_thread = Thread(target=watchdog_loop, daemon=True)
    detect_thread = Thread(target=detection_loop, daemon=True)

    telemetry_thread.start()
    watchdog_thread.start()
    detect_thread.start()

    print("[SYSTEM] Running - press Q to quit")

    try:
        while True:
            frame_to_show = display_frame
            if frame_to_show is not None:
                cv2.imshow("Drone Detection", frame_to_show)
                window_ready = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[SYSTEM] Quit requested")
                break

    except KeyboardInterrupt:
        print("[SYSTEM] KeyboardInterrupt")

    finally:
        shutdown_event.set()
        time.sleep(0.25)
        stop_drone()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()