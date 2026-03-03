import os
import random
import subprocess

AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "audio")

wav_files = [
    "hello1.wav",
    "hello2.wav",
    "hello3.wav"
]

selected = random.choice(wav_files)
path = os.path.join(AUDIO_FOLDER, selected)

print(f"Playing: {path}")

subprocess.run(["aplay", path])
