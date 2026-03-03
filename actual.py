import olympe
import os
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveTo, moveBy

DRONE_IP = os.environ.get("DRONE_IP","10.202.0.1")


def test_takeoff():
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    assert drone(TakeOff()).wait().success()
    print("Taking off...")

    # Fast ascent
    drone(moveBy(0, 0, 0.6, 0)).wait().success()
    print("Reached top")

    # Hover at top
    time.sleep(3)
    print("Hover complete")

    # Fast descent (increase Z delta slightly)
    drone(moveBy(0, 0, -0.7, 0)).wait().success()
    print("Reached bottom")

    # Land
    assert drone(Landing()).wait().success()
   
    drone.disconnect()

#tested and worked

if __name__ == "__main__":
    test_takeoff()