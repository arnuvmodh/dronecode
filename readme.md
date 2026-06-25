sudo systemctl start firmwared.service

sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi_ai.drone"::firmware="https://firmware.parrot.com/Versions/anafi2/pc/%23latest/images/anafi2-pc.ext2.zip"

sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"

terminal 1


parrot-ue4-empty


terminal 2

terminal 3


ARNUV LOOK HERE
TO RUN DRONE CODE, TYPE IN THE FOLLOWING INTO THE LINUX TERMINAL:

cd Desktop/dronecode/
^^ this maps you to the folder where the code is

python3 thenewest.py
^^this is the most recent working code with digital twin (as of 6.11)








the one with working video feed if a person is already in frame is newest.py (as of 2/25)

reminder, you have to be connected to the drone wifi
also your current is thenewest.py for audio

if fail and get stuck in air copy this into the terminal:

python3 emergency.py



for mounting the stuffs:

ip addr show enp1s0
^^check for ip address -> should see inet 192.168.50.3 or something like that

ping 192.168.50.3
^^check to see if it works

mkdir -p ~/unreal_share

sudo mount -t cifs //192.168.50.3/unrealdata ~/unreal_share \
> -o username=Kareem,uid=$(id -u),gid=$(id -g),vers=3.0

^these last two commands mount the stuff