# takeoff.py

import olympe
import os
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

# Use DRONE_IP from environment if set, otherwise default to Sphinx default
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")


def test_takeoff(on_apex=None):
    """
    Simple demo flight sequence:

    - Connect to the drone
    - Take off
    - Hover briefly at the "apex"
    - If provided, call on_apex() (e.g. play audio) at the top
    - Hover a bit longer
    - Land
    - Disconnect
    """

    print(f"[DRONE] Connecting to {DRONE_IP}...")
    drone = olympe.Drone(DRONE_IP)
    drone.connect()

    print("[DRONE] Taking off...")
    assert drone(TakeOff()).wait().success()
    print("[DRONE] Takeoff successful. Climbing / stabilizing...")

    # Let drone reach and stabilize at its default altitude
    time.sleep(3)  # treat this as the "top" of the flight

    # APEX: call audio callback here if provided
    if on_apex is not None:
        print("[DRONE] At apex → calling on_apex()...")
        on_apex()

    # Hover a bit after the audio
    time.sleep(5)

    print("[DRONE] Landing...")
    assert drone(Landing()).wait().success()
    print("[DRONE] Landed successfully.")

    drone.disconnect()
    print("[DRONE] Disconnected.")


if __name__ == "__main__":
    # Quick standalone test (no audio)
    test_takeoff()
