sudo systemctl start firmwared.service

sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi_ai.drone"::firmware="https://firmware.parrot.com/Versions/anafi2/pc/%23latest/images/anafi2-pc.ext2.zip"

sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"

terminal 1


parrot-ue4-empty


terminal 2


terminal 3

cd Desktop/dronecode/
python3 thenewest.py
the one with working video feed if a person is already in frame is newest.py (as of 2/25)

reminder, you have to be connected to the drone wifi
also your current is thenewest.py for audio

if fail and get stuck in air copy this into the terminal:

ps aux | grep python
kill -9 <PID>
