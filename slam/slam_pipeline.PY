import cv2
import numpy as np

from slam.posegraph import (
    build_pose_graph,
    add_loop_closure,
    optimize_pose_graph,
    extract_optimized_global_poses,
)
from slam.loopclosure import LoopClosureDetector


def run_slam(frames, K, cfg):
    slam_cfg = cfg["slam"]
    lc_cfg = cfg["loop_closure"]

    orb_features = slam_cfg["orb_features"]
    ransac_thresh = slam_cfg["ransac_thresh"]
    kf_interval = slam_cfg["keyframe_interval"]

    orb = cv2.ORB_create(orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    lcd = LoopClosureDetector(
    similarity_threshold=lc_cfg["similarity_threshold"],
    min_frame_gap=lc_cfg["min_frame_gap"],
    max_translation=lc_cfg.get("max_translation", 5.0)

    )

    global_poses = []
    T_global = np.eye(4)
    global_poses.append(T_global.copy())

    _, rgb0, _ = frames[0]
    prev_gray = cv2.cvtColor(rgb0, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

    lcd.add_keyframe(0, prev_gray)

    for i in range(1, len(frames)):
        _, rgb, _ = frames[i]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = orb.detectAndCompute(gray, None)

        if des_prev is None or des_curr is None:
            global_poses.append(T_global.copy())
            kp_prev, des_prev = kp_curr, des_curr
            continue

        matches = bf.knnMatch(des_prev, des_curr, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            global_poses.append(T_global.copy())
            kp_prev, des_prev = kp_curr, des_curr
            continue

        pts_prev = np.float32(
            [kp_prev[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        pts_curr = np.float32(
            [kp_curr[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        E, _ = cv2.findEssentialMat(
            pts_prev, pts_curr, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh
        )

        if E is None:
            global_poses.append(T_global.copy())
            kp_prev, des_prev = kp_curr, des_curr
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, K)

        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.squeeze()

        T_global = T_global @ np.linalg.inv(T_rel)
        global_poses.append(T_global.copy())

        if i % kf_interval == 0:
            lcd.add_keyframe(i, gray)
            lcd.detect(i, gray)

        kp_prev, des_prev = kp_curr, des_curr

    return global_poses, lcd
