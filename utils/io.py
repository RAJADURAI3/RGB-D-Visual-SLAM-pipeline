import os
import cv2
import yaml
import numpy as np

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_intrinsics(cfg):
    cam = cfg["camera"]
    return np.array([
        [cam["fx"],    0.0, cam["cx"]],
        [   0.0, cam["fy"], cam["cy"]],
        [   0.0,    0.0,       1.0]
    ], dtype=np.float64)


def load_tum_dataset(dataset_path, max_frames=300):
    def read_assoc_file(file_path):
        entries = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                ts, rel_path = line.strip().split()
                abs_path = os.path.join(dataset_path, rel_path)
                entries.append((float(ts), abs_path))
        return entries

    rgb_entries = read_assoc_file(os.path.join(dataset_path, "rgb.txt"))
    depth_entries = read_assoc_file(os.path.join(dataset_path, "depth.txt"))

    frames = []
    for i in range(min(len(rgb_entries), len(depth_entries), max_frames)):
        _, rgb_path = rgb_entries[i]
        _, depth_path = depth_entries[i]

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is not None and depth is not None:
            frames.append((i, rgb, depth))

    return frames
