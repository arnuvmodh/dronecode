import time
import traceback

import cv2
import numpy as np
import olympe


# ================= CONFIG =================
DRONE_IP = "192.168.42.1"
DISPLAY_W = 960
DISPLAY_H = 720
FRAME_TIMEOUT_SECONDS = 3.0

# Face detection settings
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
DETECTION_MIN_SIZE = (40, 40)

# Try these conversion modes in order to avoid black feed
YUV_CONVERSIONS = [
    ("NV12", cv2.COLOR_YUV2BGR_NV12),
    ("I420", cv2.COLOR_YUV2BGR_I420),
    ("YV12", cv2.COLOR_YUV2BGR_YV12),
]


# ================= STATE =================
current_frame = None
last_frame_time = 0.0
last_conversion_used = "None"
frame_counter = 0
frame_errors = 0
stream_started = False

head_detected = False
head_count = 0
best_head_center = None
best_head_size = None

drone = olympe.AnafiUSA(DRONE_IP)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ================= HELPERS =================
def frame_is_fresh():
    return (time.time() - last_frame_time) <= FRAME_TIMEOUT_SECONDS


def try_convert_yuv_to_bgr(yuv_frame):
    info = yuv_frame.info()
    h = info["raw"]["frame"]["info"]["height"]
    w = info["raw"]["frame"]["info"]["width"]

    arr = np.array(yuv_frame.as_ndarray(), copy=True)

    reshaped_candidates = []

    try:
        reshaped_candidates.append(arr.reshape((h * 3 // 2, w)))
    except Exception:
        pass

    if len(arr.shape) == 2:
        reshaped_candidates.append(arr)

    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return arr, "Already-BGR-like"

    for candidate in reshaped_candidates:
        for name, code in YUV_CONVERSIONS:
            try:
                bgr = cv2.cvtColor(candidate, code)
                if bgr is None or len(bgr.shape) != 3 or bgr.shape[2] != 3:
                    continue
                return bgr, name
            except Exception:
                continue

    raise RuntimeError(
        f"Could not convert frame. Raw ndarray shape: {arr.shape}, expected width={w}, height={h}"
    )


def build_waiting_frame():
    frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for video frames...",
        (220, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "If this stays here, check terminal output.",
        (200, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return frame


def build_status_panel():
    panel = np.zeros((320, 700, 3), dtype=np.uint8)

    lines = [
        "Anafi USA Head Detection Test",
        f"Stream started: {'YES' if stream_started else 'NO'}",
        f"Frames received: {frame_counter}",
        f"Frame fresh: {'YES' if frame_is_fresh() else 'NO'}",
        f"Last conversion: {last_conversion_used}",
        f"Frame callback errors: {frame_errors}",
        f"Head detected: {'YES' if head_detected else 'NO'}",
        f"Head count: {head_count}",
        "Mode: NO FLIGHT",
        "Press Q to quit",
    ]

    y = 35
    for i, line in enumerate(lines):
        scale = 0.85 if i == 0 else 0.72
        color = (255, 255, 255)

        if "YES" in line:
            color = (0, 255, 0)
        elif "NO" in line:
            color = (0, 200, 255)
        if "NO FLIGHT" in line:
            color = (255, 255, 0)

        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28

    if best_head_center is not None:
        cx, cy = best_head_center
        cv2.putText(
            panel,
            f"Best head center: ({cx}, {cy})",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return panel


def draw_overlay(frame, faces):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (650, 155), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    lines = [
        f"VIDEO: {'OK' if current_frame is not None else 'WAITING'}",
        f"FRESH: {'YES' if frame_is_fresh() else 'NO'}",
        f"HEAD DETECTED: {'YES' if len(faces) > 0 else 'NO'}",
        f"HEAD COUNT: {len(faces)}",
        "MODE: NO FLIGHT TEST",
    ]

    y = 35
    for line in lines:
        color = (0, 255, 0)
        if "NO" in line or "WAITING" in line:
            color = (0, 200, 255)
        if "NO FLIGHT" in line:
            color = (255, 255, 0)

        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 25

    return frame


# ================= VIDEO CALLBACK =================
def on_raw_frame(yuv_frame):
    global current_frame, last_frame_time, last_conversion_used, frame_counter, frame_errors

    try:
        bgr, conversion_name = try_convert_yuv_to_bgr(yuv_frame)
        current_frame = bgr
        last_frame_time = time.time()
        last_conversion_used = conversion_name
        frame_counter += 1

        if frame_counter % 30 == 0:
            print(f"[VIDEO] Received {frame_counter} frames. Using conversion: {conversion_name}")

    except Exception as e:
        frame_errors += 1
        print(f"[VIDEO] Frame callback error #{frame_errors}: {e}")


# ================= DETECTION =================
def detect_heads_and_draw(frame):
    global head_detected, head_count, best_head_center, best_head_size

    output = frame.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=DETECTION_SCALE_FACTOR,
        minNeighbors=DETECTION_MIN_NEIGHBORS,
        minSize=DETECTION_MIN_SIZE,
    )

    head_detected = len(faces) > 0
    head_count = len(faces)
    best_head_center = None
    best_head_size = None

    largest_area = -1

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        area = w * h

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            output,
            f"head center=({cx},{cy}) size=({w}x{h})",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if area > largest_area:
            largest_area = area
            best_head_center = (cx, cy)
            best_head_size = (w, h)

    output = draw_overlay(output, faces)

    if best_head_center is not None:
        cx, cy = best_head_center
        cv2.line(output, (cx, 0), (cx, output.shape[0]), (255, 0, 0), 1)
        cv2.line(output, (0, cy), (output.shape[1], cy), (255, 0, 0), 1)

    return output


# ================= MAIN =================
def main():
    global stream_started

    print(f"[DRONE] Connecting to {DRONE_IP} ...")
    drone.connect()
    print("[DRONE] Connected")

    try:
        print("[STREAM] Registering raw callback...")
        drone.streaming.set_callbacks(raw_cb=on_raw_frame)

        print("[STREAM] Starting stream...")
        drone.streaming.start()
        stream_started = True
        print("[STREAM] Stream start requested")

        time.sleep(2.0)

        print("[SYSTEM] Showing windows. Press Q to quit.")

        while True:
            if current_frame is not None and frame_is_fresh():
                frame_to_process = cv2.resize(current_frame, (DISPLAY_W, DISPLAY_H))
                frame_to_show = detect_heads_and_draw(frame_to_process)
            else:
                frame_to_show = build_waiting_frame()

            status_panel = build_status_panel()

            cv2.imshow("Anafi Head Detection Test", frame_to_show)
            cv2.imshow("Anafi Diagnostics", status_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[SYSTEM] Quit requested")
                break

    except KeyboardInterrupt:
        print("[SYSTEM] Keyboard interrupt")

    except Exception as e:
        print(f"[SYSTEM] Fatal error: {e}")
        traceback.print_exc()

    finally:
        print("[CLEANUP] Stopping stream and disconnecting...")

        try:
            drone.streaming.stop()
            print("[CLEANUP] Stream stopped")
        except Exception as e:
            print(f"[CLEANUP] stream.stop warning: {e}")

        try:
            drone.disconnect()
            print("[CLEANUP] Drone disconnected")
        except Exception as e:
            print(f"[CLEANUP] disconnect warning: {e}")

        cv2.destroyAllWindows()
        print("[CLEANUP] Windows closed")


if __name__ == "__main__":
    main()