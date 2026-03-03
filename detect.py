import olympe
from olympe.messages.ardrone3.Piloting import landing

DRONE_IP = "192.168.42.1"
drone = olympe.Drone(DRONE_IP)
drone.connect()
drone(Landing()).wait()
drone.disconnect()