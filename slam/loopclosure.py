import cv2
import numpy as np


class LoopClosureDetector:
    """
    Robust Loop Closure Detection:
    - ORB + KNN matching + ratio test
    - geometric verification (Essential Matrix)
    - transform validation
    """

    def __init__(self,
                 similarity_threshold=0.2,
                 min_frame_gap=20,
                 max_translation=5.0):

        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.keyframes = {}  # frame_idx -> (kp, des, image)

        self.similarity_threshold = similarity_threshold
        self.min_frame_gap = min_frame_gap
        self.max_translation = max_translation

        self.loop_closures_detected = []

    # -----------------------------------------------
    # Add Keyframe
    # -----------------------------------------------
    def add_keyframe(self, frame_idx, gray_image):
        kp, des = self.orb.detectAndCompute(gray_image, None)

        if des is not None:
            self.keyframes[frame_idx] = (kp, des, gray_image.copy())

    # -----------------------------------------------
    # Detect Loop Closure
    # -----------------------------------------------
    def detect(self, current_idx, current_gray):
        kp_curr, des_curr = self.orb.detectAndCompute(current_gray, None)

        if des_curr is None:
            return None

        best_idx = None
        best_score = 0

        for kf_idx, (kp_kf, des_kf, _) in self.keyframes.items():

            if current_idx - kf_idx < self.min_frame_gap:
                continue

            # KNN matching
            matches = self.bf.knnMatch(des_kf, des_curr, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) < 15:
                continue

            score = len(good) / len(matches)

            if score > best_score:
                best_score = score
                best_idx = kf_idx

        if best_score > self.similarity_threshold and best_idx is not None:
            print(f"[LoopClosure] {current_idx} ↔ {best_idx} | score={best_score:.3f}")

            self.loop_closures_detected.append(
                (current_idx, best_idx, best_score)
            )

            return best_idx, best_score

        return None

    # -----------------------------------------------
    # Compute Relative Transform
    # -----------------------------------------------
    def compute_relative_transform(self, idx_i, idx_j, K):

        if idx_i not in self.keyframes or idx_j not in self.keyframes:
            return None

        kp_i, des_i, img_i = self.keyframes[idx_i]
        kp_j, des_j, img_j = self.keyframes[idx_j]

        if des_i is None or des_j is None:
            return None

        matches = self.bf.knnMatch(des_i, des_j, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 15:
            return None

        pts_i = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_j = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None

        # Use only inliers
        pts_i = pts_i[mask.ravel() == 1]
        pts_j = pts_j[mask.ravel() == 1]

        if len(pts_i) < 8:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts_i, pts_j, K)

        # Validate transform
        if np.linalg.norm(t) > self.max_translation:
            return None

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()

        return T

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------
    def summary(self):
        print("\n[Loop Closure Summary]")
        print(f"Keyframes: {len(self.keyframes)}")
        print(f"Loop closures: {len(self.loop_closures_detected)}")

        for (ci, ki, sc) in self.loop_closures_detected:
            print(f"{ci} ↔ {ki} | score={sc:.3f}")
