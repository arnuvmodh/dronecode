import olympe
import os
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD, moveBy, moveTo

DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")

TARGET_ALTITUDE = 0.6  # meters
ASCENT_SPEED = 0.5     # m/s
DESCENT_SPEED = 0.5    # m/s
HOVER_TIME = 3         # seconds
CONTROL_INTERVAL = 0.05 # 50 ms between commands

def test_takeoff_target_altitude():
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    assert drone(TakeOff()).wait().success()
    print("🚁 Taking off...")

    # Fast ascent to target altitude
    current_alt = 0.0
    while current_alt < TARGET_ALTITUDE:
        drone(PCMD(0, 0, 0, ASCENT_SPEED, 0)).wait().success()
        time.sleep(CONTROL_INTERVAL)
        current_alt += ASCENT_SPEED * CONTROL_INTERVAL

    # Stop ascending
    drone(PCMD(0, 0, 0, 0, 0)).wait().success()
    print(f"⬆️ Reached target altitude: {TARGET_ALTITUDE} m")

    # Hover at target
    time.sleep(HOVER_TIME)
    print(f"⏱ Hovered for {HOVER_TIME} seconds")

    # Fast descent to ground
    current_alt = TARGET_ALTITUDE
    while current_alt > 0:
        drone(PCMD(0, 0, 0, -DESCENT_SPEED, 0)).wait().success()
        time.sleep(CONTROL_INTERVAL)
        current_alt -= DESCENT_SPEED * CONTROL_INTERVAL

    # Stop movement and land
    drone(PCMD(0, 0, 0, 0, 0)).wait().success()
    assert drone(Landing()).wait().success()
    print("🛬 Landed")

    drone.disconnect()
    print("✅ Flight finished")

if __name__ == "__main__":
    test_takeoff_target_altitude()
    