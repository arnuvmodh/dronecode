import csv
import os
import time
import olympe

from olympe.messages.ardrone3.PilotingState import AltitudeChanged

DRONE_IP="192.168.42.1"
CSV_FILE="height.csv"
TEMP_FILE=CSV_FILE+".tmp"

def write_height_atomic(height_m:float):
    with open(TEMP_FILE, "w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["height_m"])
        writer.writerow([round(height_m, 3)])
        f.flush()
        os.fsync(f.fileno())
    
    os.replace(TEMP_FILE, CSV_FILE)


def main():
    drone=olympe.AnafiUSA(DRONE_IP)
    drone.connect

    try:
        while True:
            try:
                state = drone.get_state(AltitudeChanged)
                height_m = state["altitude"]

                write_height_atomic(height_m)
                print(f"Height:{height_m:.3f}m")

    except KeyboardInterrupt:
        print("Stopping logger...")

    finally:
        drone.disconnect()

if __name__== "__main__":
    main()
