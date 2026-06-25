import os
import csv

OUT_DIR = "/home/charity/unreal_share"
TMP_FILE = os.path.join(OUT_DIR, "positions.tmp")
FINAL_FILE = os.path.join(OUT_DIR, "positions.csv")

rows = [
    ["id", "x", "y", "z"],
    ["drone1", 167, 28, 67],
]

with open(TMP_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

os.replace(TMP_FILE, FINAL_FILE)