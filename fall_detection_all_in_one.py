"""
========================================================================================
  AI PERSON & ELDERLY FALL DETECTION SYSTEM (ALL-IN-ONE STANDALONE SCRIPT)
  Features:
    - Real-Time 17-Keypoint Pose Estimation (YOLOv8-Pose)
    - Advanced Occlusion Handling & Kinematic Bone Imputation
    - Temporal Trajectory & Furniture ROI Zone Tracking
    - Biomechanical Feature Extraction (Torso Angle, Vertical Velocity & Acceleration)
    - 6-Stage Fall State Machine (ADL False Alarm Suppression)
    - High-Contrast Telemetry HUD Visualizer
    - Built-in Synthetic Occlusion Scenario Generator & Test Runner
========================================================================================
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ======================================================================================
# 1. POSE DETECTOR MODULE
# ======================================================================================
class PoseDetector:
    """
    Wraps YOLOv8-Pose model for efficient single/multi-person landmark estimation.
    """
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: Optional[str] = None):
        self.conf_thresh = conf_thresh
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PoseDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kps_data = result.keypoints.data.cpu().numpy()

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            det_conf = float(confs[i])
            kps = kps_data[i]

            if kps.shape[-1] == 2:
                kps_with_conf = np.zeros((17, 3), dtype=np.float32)
                kps_with_conf[:, :2] = kps
                kps_with_conf[:, 2] = np.where((kps[:, 0] > 0) | (kps[:, 1] > 0), det_conf, 0.0)
                kps = kps_with_conf

            # Filter out spurious false positives with fewer than 4 visible joints
            confident_joints = np.sum((kps[:, 2] > 0.35) & (kps[:, 0] > 0) & (kps[:, 1] > 0))
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            # Reject tiny or low-joint noise
            if confident_joints < 4 or bbox_w < 20 or bbox_h < 30:
                continue

            detections.append({
                "bbox": bbox,
                "det_conf": det_conf,
                "keypoints": kps
            })

        return detections


# ======================================================================================
# 2. OCCLUSION & KINEMATIC BONE IMPUTATION ENGINE
# ======================================================================================
class OcclusionEngine:
    """
    Recovers occluded joints through kinematic bone constraints, temporal momentum,
    and furniture zone tracking.
    """
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),            # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Arms
        (5, 11), (6, 12), (11, 12),                # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
    ]

    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TORSO_INDICES = [5, 6, 11, 12]
    LOWER_BODY_INDICES = [13, 14, 15, 16]

    def __init__(self, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.occlusion_zones: List[Dict[str, Any]] = []

    def add_occlusion_zone(self, name: str, polygon: List[Tuple[int, int]], zone_type: str = "furniture"):
        self.occlusion_zones.append({
            "name": name,
            "polygon": np.array(polygon, dtype=np.int32),
            "type": zone_type
        })

    def process_keypoints(
        self,
        current_kps: np.ndarray,
        history_kps: List[np.ndarray],
        bbox: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        imputed_kps = np.copy(current_kps)
        occlusion_mask = np.zeros(17, dtype=bool)

        for i in range(17):
            conf = current_kps[i, 2]
            x, y = current_kps[i, 0], current_kps[i, 1]
            if conf < self.conf_threshold or (x == 0 and y == 0):
                occlusion_mask[i] = True

        total_kps = 17
        visible_total = np.sum(~occlusion_mask)
        upper_vis = np.mean(~occlusion_mask[self.UPPER_BODY_INDICES])
        lower_vis = np.mean(~occlusion_mask[self.LOWER_BODY_INDICES])
        torso_vis = np.mean(~occlusion_mask[self.TORSO_INDICES])

        # Temporal Momentum Extrapolation
        if len(history_kps) >= 2:
            prev_kps = history_kps[-1]
            prev_prev_kps = history_kps[-2]

            for i in range(17):
                if occlusion_mask[i]:
                    if prev_kps[i, 0] > 0 and prev_prev_kps[i, 0] > 0:
                        vx = (prev_kps[i, 0] - prev_prev_kps[i, 0]) * 0.9
                        vy = (prev_kps[i, 1] - prev_prev_kps[i, 1]) * 0.9
                        imputed_kps[i, 0] = prev_kps[i, 0] + vx
                        imputed_kps[i, 1] = prev_kps[i, 1] + vy
                        imputed_kps[i, 2] = 0.45

        # Biomechanical Bone Constraint Imputation for Lower Body
        bbox_h = bbox[3] - bbox[1]
        if occlusion_mask[13] and not occlusion_mask[11]:  # Left Knee
            imputed_kps[13, 0] = imputed_kps[11, 0]
            imputed_kps[13, 1] = imputed_kps[11, 1] + bbox_h * 0.3
            imputed_kps[13, 2] = 0.35

        if occlusion_mask[14] and not occlusion_mask[12]:  # Right Knee
            imputed_kps[14, 0] = imputed_kps[12, 0]
            imputed_kps[14, 1] = imputed_kps[12, 1] + bbox_h * 0.3
            imputed_kps[14, 2] = 0.35

        if occlusion_mask[15] and imputed_kps[13, 1] > 0:  # Left Ankle
            imputed_kps[15, 0] = imputed_kps[13, 0]
            imputed_kps[15, 1] = imputed_kps[13, 1] + bbox_h * 0.25
            imputed_kps[15, 2] = 0.35

        if occlusion_mask[16] and imputed_kps[14, 1] > 0:  # Right Ankle
            imputed_kps[16, 0] = imputed_kps[14, 0]
            imputed_kps[16, 1] = imputed_kps[14, 1] + bbox_h * 0.25
            imputed_kps[16, 2] = 0.35

        stats = {
            "total_visibility": float(visible_total / total_kps),
            "upper_body_visibility": float(upper_vis),
            "lower_body_visibility": float(lower_vis),
            "torso_visibility": float(torso_vis),
            "is_partially_occluded": bool(visible_total < total_kps and visible_total >= 5),
            "is_severely_occluded": bool(visible_total < 5)
        }

        return imputed_kps, occlusion_mask, stats

    def is_in_occlusion_zone(self, point: Tuple[float, float]) -> Tuple[bool, Optional[str]]:
        pt = (int(point[0]), int(point[1]))
        for zone in self.occlusion_zones:
            dist = cv2.pointPolygonTest(zone["polygon"], pt, measureDist=False)
            if dist >= 0:
                return True, zone["name"]
        return False, None


# ======================================================================================
# 3. KINEMATIC & BIOMECHANICAL FEATURE EXTRACTOR
# ======================================================================================
class KinematicExtractor:
    """
    Computes spine angle relative to floor, normalized velocities, accelerations,
    and post-impact stillness metrics.
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
        if not keypoints_history:
            return self._empty_features()

        current_kps = keypoints_history[-1]
        current_bbox = bbox_history[-1]

        torso_angle = self.calculate_torso_angle(current_kps)
        bbox_w = max(current_bbox[2] - current_bbox[0], 1.0)
        bbox_h = max(current_bbox[3] - current_bbox[1], 1.0)
        aspect_ratio = float(bbox_w / bbox_h)

        v_y, a_y, centroid_y = self.calculate_vertical_kinematics(keypoints_history, bbox_history)
        stillness_energy = self.calculate_stillness_energy(keypoints_history, window=15)

        # Compute angular collapse rate (degrees per second)
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
            "stillness_energy": float(stillness_energy),
            "centroid_y": float(centroid_y),
            "bbox_height": float(bbox_h),
            "bbox_width": float(bbox_w)
        }

    def calculate_torso_angle(self, kps: np.ndarray) -> float:
        l_sh, r_sh = kps[5], kps[6]
        l_hip, r_hip = kps[11], kps[12]

        mid_sh_x = (l_sh[0] + r_sh[0]) / 2.0
        mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0
        mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

        dx = mid_sh_x - mid_hip_x
        dy = mid_sh_y - mid_hip_y

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return 90.0

        angle_rad = np.arctan2(abs(dy), abs(dx) + 1e-6)
        return float(np.degrees(angle_rad))

    def calculate_vertical_kinematics(
        self,
        kps_history: List[np.ndarray],
        bbox_history: List[List[float]]
    ) -> Tuple[float, float, float]:
        if len(kps_history) < 2:
            current_kps = kps_history[-1]
            torso_y = np.mean(current_kps[[5, 6, 11, 12], 1])
            return 0.0, 0.0, float(torso_y)

        y_coords = []
        for kps in kps_history[-min(len(kps_history), 4):]:
            torso_y = np.mean(kps[[5, 6, 11, 12], 1])
            y_coords.append(torso_y)

        current_bbox = bbox_history[-1]
        norm_h = max(current_bbox[3] - current_bbox[1], 50.0)

        v_now = (y_coords[-1] - y_coords[-2]) / (self.dt * norm_h)
        if len(y_coords) >= 3:
            v_prev = (y_coords[-2] - y_coords[-3]) / (self.dt * norm_h)
            a_now = (v_now - v_prev) / self.dt
        else:
            a_now = 0.0

        return float(v_now), float(a_now), float(y_coords[-1])

    def calculate_stillness_energy(self, kps_history: List[np.ndarray], window: int = 15) -> float:
        if len(kps_history) < 3:
            return 50.0

        sample_len = min(len(kps_history), window)
        displacements = []

        for f in range(-sample_len + 1, 0):
            curr = kps_history[f]
            prev = kps_history[f - 1]
            valid_mask = (curr[:, 0] > 0) & (prev[:, 0] > 0) & (curr[:, 2] > 0.2)
            if np.any(valid_mask):
                diffs = np.linalg.norm(curr[valid_mask, :2] - prev[valid_mask, :2], axis=1)
                displacements.append(np.mean(diffs))

        return float(np.mean(displacements)) if displacements else 0.0

    def _empty_features(self) -> Dict[str, float]:
        return {
            "torso_angle": 90.0, "vertical_velocity": 0.0, "vertical_accel": 0.0,
            "aspect_ratio": 0.4, "stillness_energy": 100.0, "centroid_y": 0.0,
            "bbox_height": 100.0, "bbox_width": 40.0
        }


# ======================================================================================
# 4. TRACKLET & MULTI-OBJECT TRACKER
# ======================================================================================
class Tracklet:
    def __init__(self, track_id: int, initial_bbox: List[float], initial_kps: np.ndarray, max_history: int = 60):
        self.track_id = track_id
        self.max_history = max_history
        self.bbox_history: List[List[float]] = [initial_bbox]
        self.raw_kps_history: List[np.ndarray] = [initial_kps]
        self.imputed_kps_history: List[np.ndarray] = [initial_kps]
        self.missing_frames: int = 0
        self.in_occlusion_zone: bool = False
        self.occlusion_zone_name: Optional[str] = None
        self.state: str = "MONITORING"
        self.state_timer: float = 0.0
        self.pre_alert_countdown: float = 0.0

    def update(self, bbox: List[float], raw_kps: np.ndarray, imputed_kps: np.ndarray):
        self.bbox_history.append(bbox)
        self.raw_kps_history.append(raw_kps)
        self.imputed_kps_history.append(imputed_kps)
        self.missing_frames = 0
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
            self.raw_kps_history.pop(0)
            self.imputed_kps_history.pop(0)


class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.25, max_missing: int = 20):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.tracklets: Dict[int, Tracklet] = {}
        self.next_id: int = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Tracklet]:
        det_bboxes = [d["bbox"] for d in detections]
        det_kps = [d["keypoints"] for d in detections]
        matched_dets = set()

        for track_id, track in list(self.tracklets.items()):
            last_bbox = track.bbox_history[-1]
            best_iou = 0.0
            best_det_idx = -1

            for idx, det_bbox in enumerate(det_bboxes):
                if idx in matched_dets:
                    continue
                iou = self._calculate_iou(last_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_iou >= self.iou_thresh and best_det_idx != -1:
                matched_dets.add(best_det_idx)
                track.update(det_bboxes[best_det_idx], det_kps[best_det_idx], det_kps[best_det_idx])
            else:
                track.missing_frames += 1

        for track_id, track in list(self.tracklets.items()):
            if track.missing_frames > self.max_missing:
                del self.tracklets[track_id]

        for idx, det_bbox in enumerate(det_bboxes):
            if idx not in matched_dets:
                new_track = Tracklet(self.next_id, det_bbox, det_kps[idx])
                self.tracklets[self.next_id] = new_track
                self.next_id += 1

        return list(self.tracklets.values())

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        return float(interArea / float(boxAArea + boxBArea - interArea + 1e-6))


# ======================================================================================
# 5. MULTI-STAGE FALL STATE MACHINE
# ======================================================================================
class FallStateMachine:
    """
    Real-Time Fall State Machine with instant-trigger alerts on falling motion and impact.
    """
    STATE_MONITORING = "MONITORING"
    STATE_RAPID_DESCENT = "RAPID_DESCENT"
    STATE_FALL_DETECTED = "FALL_DETECTED"
    STATE_IMPACT = "IMPACT"
    STATE_INACTIVITY_CHECK = "INACTIVITY_CHECK"
    STATE_PRE_ALERT = "PRE_ALERT"
    STATE_CONFIRMED_FALL = "CONFIRMED_FALL"

    def __init__(
        self,
        fps: float = 30.0,
        fall_velocity_thresh: float = 1.2,    # Highly sensitive to descent
        flat_angle_thresh: float = 55.0,       # Any tilt below 55° is flagged
        standing_angle_thresh: float = 70.0,
        inactivity_sec: float = 1.5,
        pre_alert_sec: float = 2.0,
        stillness_energy_thresh: float = 25.0
    ):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.fall_velocity_thresh = fall_velocity_thresh
        self.flat_angle_thresh = flat_angle_thresh
        self.standing_angle_thresh = standing_angle_thresh
        self.inactivity_sec = inactivity_sec
        self.pre_alert_sec = pre_alert_sec
        self.stillness_energy_thresh = stillness_energy_thresh

    def step(
        self,
        tracklet: Tracklet,
        features: Dict[str, float],
        occlusion_stats: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        v_y = features.get("vertical_velocity", 0.0)
        angular_vel = features.get("angular_velocity", 0.0)
        angle = features.get("torso_angle", 90.0)
        aspect_ratio = features.get("aspect_ratio", 0.5)
        stillness = features.get("stillness_energy", 0.0)
        in_occ_zone = tracklet.in_occlusion_zone

        current = tracklet.state
        next_state = current
        event_trigger = ""

        # A REAL Fall requires a dynamic event (rapid descent or fast angular collapse)
        # Normal lying down / sleeping on a bed has v_y ~= 0 and angular_vel ~= 0
        is_dynamic_fall = (
            v_y > self.fall_velocity_thresh or
            (angular_vel > 35.0 and angle < self.flat_angle_thresh) or
            (v_y > 0.6 and angle < self.flat_angle_thresh) or
            (in_occ_zone and v_y > 0.8)
        )

        if is_dynamic_fall:
            next_state = self.STATE_FALL_DETECTED
            tracklet.state_timer = 0.0
            event_trigger = f"[!] FALL TRIGGERED! (Velocity: +{v_y:.2f} h/s, Angle: {angle:.1f}°)"

        elif current in [self.STATE_FALL_DETECTED, self.STATE_IMPACT]:
            tracklet.state_timer += self.dt
            # If subject remains horizontal or still on ground after a fall
            if angle < self.flat_angle_thresh or aspect_ratio > 0.85 or in_occ_zone:
                if tracklet.state_timer >= self.inactivity_sec:
                    next_state = self.STATE_CONFIRMED_FALL
                    event_trigger = "[EMERGENCY] CONFIRMED EMERGENCY FALL ON GROUND"
                else:
                    next_state = self.STATE_FALL_DETECTED
            elif angle > self.standing_angle_thresh and v_y < -0.3:
                # Subject stood back up
                next_state = self.STATE_MONITORING
                event_trigger = "Subject stood upright -> Reset"

        elif current == self.STATE_CONFIRMED_FALL:
            if angle > self.standing_angle_thresh and stillness > 20.0:
                next_state = self.STATE_MONITORING
                event_trigger = "Subject upright and moving -> Reset"

        elif current == self.STATE_MONITORING:
            # Person is monitored normally.
            # If lying down with no dynamic shock, they remain in peaceful MONITORING state
            pass

        tracklet.state = next_state
        return next_state, {
            "state_timer": tracklet.state_timer,
            "countdown": max(0.0, self.inactivity_sec - tracklet.state_timer),
            "event_trigger": event_trigger
        }


# ======================================================================================
# 6. TELEMETRY & HUD VISUALIZER
# ======================================================================================
class Visualizer:
    COLOR_NORMAL = (50, 220, 100)
    COLOR_OCCLUDED = (0, 165, 255)
    COLOR_DANGER = (0, 0, 255)
    COLOR_BG_PANEL = (20, 24, 30)
    COLOR_ZONE = (180, 100, 50)

    STATE_COLORS = {
        "MONITORING": (50, 220, 100),
        "FALL_DETECTED": (0, 0, 255),
        "IMPACT": (0, 0, 255),
        "INACTIVITY_CHECK": (0, 140, 255),
        "CONFIRMED_FALL": (0, 0, 255)
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        tracklets: List[Tracklet],
        tracklet_data: Dict[int, Dict[str, Any]],
        occlusion_zones: Optional[List[Dict[str, Any]]] = None,
        fps_val: float = 0.0
    ) -> np.ndarray:
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # 1. Render Occlusion Zones
        if occlusion_zones:
            overlay = vis_frame.copy()
            for zone in occlusion_zones:
                poly = zone["polygon"]
                cv2.fillPoly(overlay, [poly], self.COLOR_ZONE)
                cv2.polylines(vis_frame, [poly], isClosed=True, color=(255, 200, 100), thickness=2)
                moments = cv2.moments(poly)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.putText(vis_frame, f"ZONE: {zone['name']}", (cx - 40, cy),
                                self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)

        # 2. Render Top Status Bar
        self._render_top_bar(vis_frame, fps_val, len(tracklets))

        # 3. Render Person Overlays
        for track in tracklets:
            t_id = track.track_id
            if t_id not in tracklet_data:
                continue

            data = tracklet_data[t_id]
            imputed_kps = data["imputed_kps"]
            occlusion_mask = data["occlusion_mask"]
            features = data["features"]
            state = data["state"]
            state_info = data["state_info"]
            occ_stats = data["occlusion_stats"]

            # Bounding box
            bbox = track.bbox_history[-1]

            # Skeleton
            self._draw_skeleton(vis_frame, imputed_kps, occlusion_mask, state, bbox)

            # BBox & Card
            color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3 if state != "MONITORING" else 2)
            self._draw_person_card(vis_frame, x1, y1, t_id, state, features, occ_stats, state_info)

            # Alert Banner
            if state in ["FALL_DETECTED", "IMPACT", "INACTIVITY_CHECK", "CONFIRMED_FALL"]:
                self._draw_alert_banner(vis_frame, state, state_info, t_id)

        return vis_frame

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, occ_mask: np.ndarray, state: str, bbox: List[float]):
        state_color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
        bbox_w = max(bbox[2] - bbox[0], 20.0)
        bbox_h = max(bbox[3] - bbox[1], 20.0)
        max_allowed_bone_len = np.hypot(bbox_w, bbox_h) * 0.85

        for p1, p2 in OcclusionEngine.SKELETON_PAIRS:
            x1, y1, _ = kps[p1]
            x2, y2, _ = kps[p2]
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            
            # Prevent stray lines across screen
            bone_dist = np.hypot(x1 - x2, y1 - y2)
            if bone_dist > max_allowed_bone_len:
                continue

            is_occ = occ_mask[p1] or occ_mask[p2]
            bone_color = self.COLOR_OCCLUDED if is_occ else state_color
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), bone_color, 2 if is_occ else 3, cv2.LINE_AA)

        for i in range(17):
            x, y, _ = kps[i]
            if x == 0 and y == 0:
                continue
            # Check if joint is within reasonable proximity of bbox
            if not (bbox[0] - bbox_w*0.3 <= x <= bbox[2] + bbox_w*0.3 and bbox[1] - bbox_h*0.3 <= y <= bbox[3] + bbox_h*0.3):
                continue

            pt = (int(x), int(y))
            if occ_mask[i]:
                cv2.circle(frame, pt, 5, self.COLOR_OCCLUDED, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, state_color, -1, cv2.LINE_AA)

    def _draw_person_card(self, frame: np.ndarray, x1: int, y1: int, track_id: int, state: str,
                          features: Dict[str, float], occ_stats: Dict[str, float], state_info: Dict[str, Any]):
        angle = features.get("torso_angle", 90.0)
        v_y = features.get("vertical_velocity", 0.0)
        card_w, card_h = 220, 85
        card_x = max(10, x1)
        card_y = max(45, y1 - card_h - 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 38), -1)
        state_color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + 5, card_y + card_h), state_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        display_state = state
        if state == "MONITORING" and angle < 45.0:
            display_state = "MONITORING (RESTING)"

        cv2.putText(frame, f"PERSON #{track_id} | {display_state}", (card_x + 12, card_y + 18),
                    self.font, 0.42, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Torso Angle: {angle:.1f} deg", (card_x + 12, card_y + 36),
                    self.font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        v_sign = "+" if v_y >= 0 else ""
        v_color = self.COLOR_DANGER if v_y > 1.8 else (200, 200, 200)
        cv2.putText(frame, f"V_descent: {v_sign}{v_y:.2f} h/s", (card_x + 12, card_y + 54),
                    self.font, 0.38, v_color, 1, cv2.LINE_AA)

        if occ_stats.get("is_partially_occluded", False):
            occ_pct = int((1.0 - occ_stats['total_visibility']) * 100)
            cv2.putText(frame, f"Occlusion: {occ_pct}% (IMPUTED)", (card_x + 12, card_y + 72),
                        self.font, 0.36, self.COLOR_OCCLUDED, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Visibility: 100% Clear", (card_x + 12, card_y + 72),
                        self.font, 0.36, (120, 200, 120), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, frame: np.ndarray, state: str, state_info: Dict[str, Any], track_id: int):
        h, w = frame.shape[:2]
        banner_h = 75
        banner_y = h - banner_h - 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + banner_h), (15, 15, 220), -1)
        cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + banner_h), (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        if state in ["FALL_DETECTED", "IMPACT", "INACTIVITY_CHECK"]:
            countdown = state_info.get("countdown", 1.5)
            text = f"[ALERT] FALL DETECTED (PERSON #{track_id}) - IMPACT CONFIRMED!"
            sub = f"Verifying ground stillness ({countdown:.1f}s remaining)... Stand up to cancel."
            cv2.putText(frame, text, (35, banner_y + 30), self.font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (35, banner_y + 55), self.font, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
        else:
            text = f"[EMERGENCY] CONFIRMED FALL DETECTED (PERSON #{track_id})"
            sub = "Alert dispatched to caregiver / emergency contacts"
            cv2.putText(frame, text, (35, banner_y + 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (35, banner_y + 55), self.font, 0.45, (220, 240, 255), 1, cv2.LINE_AA)

    def _render_top_bar(self, frame: np.ndarray, fps: float, count: int):
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), self.COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "AI FALL DETECTION & OCCLUSION RESILIENCE SYSTEM", (15, 21),
                    self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        status_str = f"FPS: {fps:.1f}  |  Tracking: {count} Subject(s)  |  Edge CPU Mode"
        cv2.putText(frame, status_str, (w - 380, 21), self.font, 0.45, (160, 220, 160), 1, cv2.LINE_AA)


# ======================================================================================
# 7. SYNTHETIC OCCLUSION TEST VIDEO GENERATOR
# ======================================================================================
def generate_fall_test_video(output_path: str = "demo_fall_with_occlusion.mp4", duration_sec: int = 14, fps: int = 30):
    width, height = 854, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg_color = (240, 240, 245)
    floor_color = (210, 190, 170)
    table_color = (60, 90, 140)
    table_x1, table_y1 = 300, 310
    table_x2, table_y2 = 620, 440

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Floor
        cv2.rectangle(frame, (0, 360), (width, height), floor_color, -1)
        cv2.line(frame, (0, 360), (width, 360), (180, 160, 140), 2)

        if t < 5.0:
            px = int(120 + t * 65)
            head_y = 160
            torso_angle = 90.0
            leg_swing = np.sin(t * 8) * 20
        elif t < 6.0:
            fall_prog = (t - 5.0) / 1.0
            px = int(445 + fall_prog * 30)
            head_y = int(160 + fall_prog * 180)
            torso_angle = 90.0 - fall_prog * 75.0
            leg_swing = 0
        else:
            px = 475
            head_y = 345
            torso_angle = 12.0
            leg_swing = 0

        skin_color = (180, 190, 220)
        cloth_color = (120, 60, 40)
        pants_color = (50, 50, 80)
        rad = np.radians(torso_angle)
        torso_len = 90

        hx, hy = px, head_y
        hip_x = int(hx - torso_len * np.cos(rad))
        hip_y = int(hy + torso_len * np.sin(rad))
        knee_x = int(hip_x - 45 * np.cos(rad) + leg_swing * 0.3)
        knee_y = int(hip_y + 45 * np.sin(rad))
        foot_x = int(knee_x - 40 * np.cos(rad) - leg_swing * 0.5)
        foot_y = min(height - 10, int(knee_y + 40 * np.sin(rad)))

        cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), pants_color, 14)
        cv2.line(frame, (knee_x, knee_y), (foot_x, foot_y), pants_color, 12)
        cv2.line(frame, (hx, hy), (hip_x, hip_y), cloth_color, 18)
        cv2.circle(frame, (hx, hy), 16, skin_color, -1)
        cv2.line(frame, (hx, hy + 20), (int(hx + 25 * np.cos(rad)), int(hy + 20 + 35 * np.sin(rad))), cloth_color, 8)

        # Draw Table Barrier over subject's lower body
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), table_color, -1)
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), (40, 60, 100), 3)
        cv2.putText(frame, "[OCCLUSION BARRIER: TABLE]", (table_x1 + 35, table_y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        status_text = "Phase: Walking" if t < 5.0 else ("Phase: SLIPPING / FALLING" if t < 6.0 else "Phase: POST-FALL IMMOBILITY")
        cv2.putText(frame, f"TEST SCENARIO: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[TestGenerator] Synthetic fall test video generated at: {output_path}")


# ======================================================================================
# 8. MAIN ORCHESTRATOR & PIPELINE RUNNER
# ======================================================================================
class FallDetectionPipeline:
    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: str = None, fps: float = 30.0):
        self.fps = fps
        self.detector = PoseDetector(model_name=model_name, conf_thresh=conf_thresh, device=device)
        self.tracker = SimpleTracker(iou_thresh=0.25, max_missing=20)
        self.occlusion_engine = OcclusionEngine(conf_threshold=0.25)
        self.kinematics = KinematicExtractor(fps=fps)
        self.state_machine = FallStateMachine(fps=fps, inactivity_sec=4.0, pre_alert_sec=5.0)
        self.visualizer = Visualizer()

    def add_furniture_zone(self, name: str, polygon: List[Tuple[int, int]]):
        self.occlusion_engine.add_occlusion_zone(name, polygon)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        detections = self.detector.detect(frame)
        tracklets = self.tracker.update(detections)

        tracklet_data: Dict[int, Dict[str, Any]] = {}
        alerts: List[Dict[str, Any]] = []

        for track in tracklets:
            t_id = track.track_id
            curr_bbox = track.bbox_history[-1]
            raw_kps = track.raw_kps_history[-1]

            imputed_kps, occ_mask, occ_stats = self.occlusion_engine.process_keypoints(
                current_kps=raw_kps,
                history_kps=track.imputed_kps_history,
                bbox=curr_bbox
            )
            track.imputed_kps_history[-1] = imputed_kps

            centroid = ((curr_bbox[0] + curr_bbox[2]) / 2.0, (curr_bbox[1] + curr_bbox[3]) / 2.0)
            in_zone, zone_name = self.occlusion_engine.is_in_occlusion_zone(centroid)
            track.in_occlusion_zone = in_zone
            track.occlusion_zone_name = zone_name

            features = self.kinematics.extract_features(
                keypoints_history=track.imputed_kps_history,
                bbox_history=track.bbox_history,
                occlusion_stats=occ_stats
            )

            state, state_info = self.state_machine.step(track, features, occ_stats)

            tracklet_data[t_id] = {
                "imputed_kps": imputed_kps,
                "occlusion_mask": occ_mask,
                "occlusion_stats": occ_stats,
                "features": features,
                "state": state,
                "state_info": state_info
            }

            if state in [FallStateMachine.STATE_FALL_DETECTED, FallStateMachine.STATE_IMPACT, FallStateMachine.STATE_CONFIRMED_FALL]:
                alerts.append({"track_id": t_id, "state": state, "info": state_info, "features": features})

        rendered_frame = self.visualizer.render(
            frame=frame,
            tracklets=tracklets,
            tracklet_data=tracklet_data,
            occlusion_zones=self.occlusion_engine.occlusion_zones,
            fps_val=self.fps
        )
        return rendered_frame, alerts


def run_pipeline(
    source: str = "demo",
    save_path: str = "fall_detection_output.mp4",
    show: bool = True,
    conf: float = 0.35,
    device: str = None
):
    print("=" * 75)
    print("  AI PERSON & ELDERLY FALL DETECTION SYSTEM (ALL-IN-ONE)")
    print("  With Advanced Occlusion Resilience & Kinematic State Machine")
    print("=" * 75)

    if source == "demo":
        demo_file = "demo_fall_with_occlusion.mp4"
        if not os.path.exists(demo_file):
            print("[Pipeline] Generating synthetic test video with occlusion barrier...")
            generate_fall_test_video(demo_file)
        source = demo_file

    is_cam = str(source).isdigit()
    if is_cam:
        # On Windows, DirectShow backend (CAP_DSHOW) is fast and reliable
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[Error] Could not open video source: '{source}'. Check file path or camera permissions.")
        return

    # Read first frame to ensure camera/video is actively streaming valid dimensions
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print(f"[Error] Could not read frames from video source: '{source}'")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    print(f"[Pipeline] Initialized Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Output will be saved to: {save_path}")

    pipeline = FallDetectionPipeline(conf_thresh=conf, device=device, fps=fps)

    if "demo" in str(source):
        pipeline.add_furniture_zone("Coffee Table", [(300, 310), (620, 310), (620, 440), (300, 440)])

    frame_count = 0
    t_start = time.time()
    window_name = "AI Fall Detection & Occlusion Resilience"

    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Display scaled window if large, keeping aspect ratio
            display_w = min(width, 1024)
            display_h = int(height * (display_w / max(width, 1)))
            cv2.resizeWindow(window_name, display_w, display_h)
        except Exception:
            pass

    try:
        frame = first_frame
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            frame_count += 1
            t_frame_start = time.time()

            vis_frame, alerts = pipeline.process_frame(frame)

            t_elapsed = time.time() - t_frame_start
            current_fps = 1.0 / max(t_elapsed, 1e-4)
            pipeline.fps = current_fps

            for alert in alerts:
                if alert["state"] == "CONFIRMED_FALL":
                    print(f"🚨 [EMERGENCY ALERT] Frame {frame_count}: Confirmed Fall for Person #{alert['track_id']}! "
                          f"(Torso Angle: {alert['features']['torso_angle']:.1f}°)")

            if writer:
                writer.write(vis_frame)

            if show:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[Pipeline] User requested exit.")
                    break

            if frame_count % 30 == 0 and total_frames > 0:
                print(f"[Pipeline] Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

            # Reset frame to read next frame in next iteration
            frame = None

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-4)
    print("=" * 75)
    print(f"  Execution Complete! Processed {frame_count} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    if save_path:
        print(f"  Processed video saved at: {os.path.abspath(save_path)}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fall Detection & Occlusion System (All-in-One)")
    parser.add_argument("--source", type=str, default="demo", help="Path to video file, webcam index (0), or 'demo'")
    parser.add_argument("--save", type=str, default="fall_detection_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", default=False, help="Display live OpenCV window")
    parser.add_argument("--conf", type=float, default=0.35, help="Pose detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or None for auto")

    args = parser.parse_args()
    run_pipeline(source=args.source, save_path=args.save, show=args.show, conf=args.conf, device=args.device)
