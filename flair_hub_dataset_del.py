import os
import shutil

root_dir = "D:\kanyamahanga\Datasets\FLAIR_HUB\data"  # <-- change this!

for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
    for f in filenames:
        if "_SENTINEL1" in f and f.lower().endswith(".zip"):
            full_path = os.path.join(dirpath, f)
            print(f"Deleting ZIP file: {full_path}")
            os.remove(full_path)

    for f in filenames:
        if "_ALL_LABEL-LPIS" in f and f.lower().endswith(".zip"):
            full_path = os.path.join(dirpath, f)
            print(f"Deleting ZIP file: {full_path}")
            os.remove(full_path)

    for f in filenames:
        if "_AERIAL-RLT_PAN" in f and f.lower().endswith(".zip"):
            full_path = os.path.join(dirpath, f)
            print(f"Deleting ZIP file: {full_path}")
            os.remove(full_path)
