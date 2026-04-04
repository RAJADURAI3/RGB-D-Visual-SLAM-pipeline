import numpy as np
import cv2
import time

from utils.io import load_config, get_intrinsics
from slam.slam_pipeline import run_slam


class SLAMService:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.K = get_intrinsics(self.cfg)

        self.frames = []
        self.current_pose = np.eye(4)
        self.loop_closures = 0

        self.total_time = 0
        self.frame_count = 0

    def process_frame(self, frame):
        start = time.time()

        h, w = frame.shape[:2]
        depth = np.ones((h, w), dtype=np.uint16) * 5000

        self.frames.append((0.0, frame, depth))
        window = self.frames[-20:]

        if len(window) >= 2:
            poses, lcd = run_slam(window, self.K, self.cfg)
            self.current_pose = poses[-1]
            self.loop_closures = len(lcd.loop_closures_detected)

        end = time.time()

        self.total_time += (end - start)
        self.frame_count += 1

        fps = self.frame_count / self.total_time if self.total_time > 0 else 0

        return {
            "pose": self.current_pose,
            "loop_closures": self.loop_closures,
            "frames": len(self.frames),
            "fps": round(fps, 2)
        }

    def reset(self):
        self.frames = []
        self.current_pose = np.eye(4)
        self.loop_closures = 0
        self.total_time = 0
        self.frame_count = 0
