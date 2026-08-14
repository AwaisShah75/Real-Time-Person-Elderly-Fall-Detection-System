"""
Kinematic & Biomechanical Feature Extraction Module
Computes torso angles, normalized velocities, accelerations, aspect ratios,
and post-impact stillness metrics.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class KinematicExtractor:
    """
    Extracts physically explainable biomechanical features from temporal pose sequences.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)

    def extract_features(
        self,
        keypoints_history: List[np.ndarray],
        bbox_history: List[List[float]],
        occlusion_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Computes real-time kinematic metrics from recent pose buffer.

        Args:
            keypoints_history: List of (17, 3) arrays for the past N frames.
            bbox_history: List of [x1, y1, x2, y2] bounding boxes for the past N frames.
            occlusion_stats: Current frame occlusion metrics.

        Returns:
            Dict containing:
                - torso_angle: float (0° to 90°)
                - vertical_velocity: float (normalized units/sec, positive is downward)
                - vertical_accel: float (normalized units/sec^2)
                - aspect_ratio: float (width / height)
                - aspect_ratio_delta: float
                - stillness_energy: float (average pixel motion across visible joints)
                - centroid_y: float
        """
        if not keypoints_history:
            return self._empty_features()

        current_kps = keypoints_history[-1]
        current_bbox = bbox_history[-1]

        # 1. Torso Angle
        torso_angle = self.calculate_torso_angle(current_kps)

        # 2. Aspect Ratio
        bbox_w = max(current_bbox[2] - current_bbox[0], 1.0)
        bbox_h = max(current_bbox[3] - current_bbox[1], 1.0)
        aspect_ratio = float(bbox_w / bbox_h)

        # 3. Normalized Vertical Kinematics
        v_y, a_y, centroid_y = self.calculate_vertical_kinematics(keypoints_history, bbox_history)

        # 4. Aspect Ratio Rate of Change
        if len(bbox_history) >= 2:
            prev_bbox = bbox_history[-2]
            prev_w = max(prev_bbox[2] - prev_bbox[0], 1.0)
            prev_h = max(prev_bbox[3] - prev_bbox[1], 1.0)
            prev_ar = prev_w / prev_h
            ar_delta = float((aspect_ratio - prev_ar) / self.dt)
        else:
            ar_delta = 0.0

        # 5. Stillness Energy over recent window (last 10-15 frames)
        stillness_energy = self.calculate_stillness_energy(keypoints_history, window=15)

        # 6. Angular Collapse Rate (degrees per second)
        angular_vel = 0.0
        if len(keypoints_history) >= 2:
            prev_kps = keypoints_history[-2]
            prev_angle = self.calculate_torso_angle(prev_kps)
            angular_vel = (prev_angle - torso_angle) * self.fps

        return {
            "torso_angle": float(torso_angle),
            "angular_velocity": float(angular_vel),
            "vertical_velocity": float(v_y),
            "vertical_accel": float(a_y),
            "aspect_ratio": float(aspect_ratio),
            "aspect_ratio_delta": float(ar_delta),
            "stillness_energy": float(stillness_energy),
            "centroid_y": float(centroid_y),
            "bbox_height": float(bbox_h),
            "bbox_width": float(bbox_w)
        }

    def calculate_torso_angle(self, kps: np.ndarray) -> float:
        """
        Computes the acute angle (0 to 90 degrees) of the spine vector relative to horizontal floor.
        Standing = ~75°-90°, Lying/Fallen = ~0°-35°.
        """
        # COCO: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
        l_sh, r_sh = kps[5], kps[6]
        l_hip, r_hip = kps[11], kps[12]

        mid_sh_x = (l_sh[0] + r_sh[0]) / 2.0
        mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0

        mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

        dx = mid_sh_x - mid_hip_x
        dy = mid_sh_y - mid_hip_y  # Note: screen y increases downward

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return 90.0

        # Angle relative to horizontal floor
        angle_rad = np.arctan2(abs(dy), abs(dx) + 1e-6)
        angle_deg = float(np.degrees(angle_rad))
        return angle_deg

    def calculate_vertical_kinematics(
        self,
        kps_history: List[np.ndarray],
        bbox_history: List[List[float]]
    ) -> Tuple[float, float, float]:
        """
        Computes normalized vertical velocity and acceleration of the torso centroid.
        """
        if len(kps_history) < 2:
            current_kps = kps_history[-1]
            torso_y = np.mean(current_kps[[5, 6, 11, 12], 1])
            return 0.0, 0.0, float(torso_y)

        # Get torso centroid Y coordinates for last 3 frames
        y_coords = []
        for kps in kps_history[-min(len(kps_history), 4):]:
            torso_y = np.mean(kps[[5, 6, 11, 12], 1])
            y_coords.append(torso_y)

        # Normalization factor: current bounding box height
        current_bbox = bbox_history[-1]
        norm_h = max(current_bbox[3] - current_bbox[1], 50.0)

        # Velocity in normalized heights per second
        # Positive = moving downward
        v_now = (y_coords[-1] - y_coords[-2]) / (self.dt * norm_h)

        if len(y_coords) >= 3:
            v_prev = (y_coords[-2] - y_coords[-3]) / (self.dt * norm_h)
            a_now = (v_now - v_prev) / self.dt
        else:
            a_now = 0.0

        return float(v_now), float(a_now), float(y_coords[-1])

    def calculate_stillness_energy(self, kps_history: List[np.ndarray], window: int = 15) -> float:
        """
        Computes mean pixel movement across visible joints in the recent window.
        Low value (<10.0) indicates immobility/stillness.
        """
        if len(kps_history) < 3:
            return 50.0

        sample_len = min(len(kps_history), window)
        displacements = []

        for f in range(-sample_len + 1, 0):
            curr = kps_history[f]
            prev = kps_history[f - 1]

            # Only calculate displacement for joints with valid coordinates
            valid_mask = (curr[:, 0] > 0) & (prev[:, 0] > 0) & (curr[:, 2] > 0.2)
            if np.any(valid_mask):
                diffs = np.linalg.norm(curr[valid_mask, :2] - prev[valid_mask, :2], axis=1)
                displacements.append(np.mean(diffs))

        if not displacements:
            return 0.0

        return float(np.mean(displacements))

    def _empty_features(self) -> Dict[str, float]:
        return {
            "torso_angle": 90.0,
            "vertical_velocity": 0.0,
            "vertical_accel": 0.0,
            "aspect_ratio": 0.4,
            "aspect_ratio_delta": 0.0,
            "stillness_energy": 100.0,
            "centroid_y": 0.0,
            "bbox_height": 100.0,
            "bbox_width": 40.0
        }
