"""
Visualizer Module
Renders real-time telemetry HUD, color-coded skeletons with occlusion indicators,
and alert banners over video frames.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from core.occlusion import OcclusionEngine
from core.tracker import Tracklet


class Visualizer:
    """
    Renders high-contrast, informative overlays for monitoring and debugging.
    """

    # Colors in BGR
    COLOR_NORMAL = (50, 220, 100)       # Neon Green
    COLOR_OCCLUDED = (0, 165, 255)      # Amber / Orange
    COLOR_WARNING = (0, 200, 255)       # Yellow
    COLOR_DANGER = (0, 0, 255)          # Bright Red
    COLOR_BG_PANEL = (20, 24, 30)       # Dark Slate
    COLOR_ZONE = (180, 100, 50)         # Soft Blue/Purple for Furniture ROI

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
        """
        Renders the complete telemetry and skeleton overlays.
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # 1. Render Occlusion Zones (Furniture ROIs)
        if occlusion_zones:
            overlay = vis_frame.copy()
            for zone in occlusion_zones:
                poly = zone["polygon"]
                cv2.fillPoly(overlay, [poly], self.COLOR_ZONE)
                cv2.polylines(vis_frame, [poly], isClosed=True, color=(255, 200, 100), thickness=2)
                
                # Zone label
                moments = cv2.moments(poly)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.putText(vis_frame, f"ZONE: {zone['name']}", (cx - 40, cy),
                                self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)

        # 2. Render Global Top Status Bar
        self._render_top_bar(vis_frame, fps_val, len(tracklets))

        # 3. Render Each Detected Person
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

            # Skeleton
            bbox = track.bbox_history[-1]
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

        # Draw bones
        for p1, p2 in OcclusionEngine.SKELETON_PAIRS:
            x1, y1, c1 = kps[p1]
            x2, y2, c2 = kps[p2]

            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue

            # Prevent stray lines across screen
            bone_dist = np.hypot(x1 - x2, y1 - y2)
            if bone_dist > max_allowed_bone_len:
                continue

            is_occluded_bone = occ_mask[p1] or occ_mask[p2]
            bone_color = self.COLOR_OCCLUDED if is_occluded_bone else state_color
            thickness = 2 if is_occluded_bone else 3

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), bone_color, thickness, cv2.LINE_AA)

        # Draw joints
        for i in range(17):
            x, y, conf = kps[i]
            if x == 0 and y == 0:
                continue

            # Check if joint is within reasonable proximity of bbox
            if not (bbox[0] - bbox_w*0.3 <= x <= bbox[2] + bbox_w*0.3 and bbox[1] - bbox_h*0.3 <= y <= bbox[3] + bbox_h*0.3):
                continue

            pt = (int(x), int(y))
            if occ_mask[i]:
                # Imputed joint: Orange hollow circle
                cv2.circle(frame, pt, 5, self.COLOR_OCCLUDED, 2, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Confident joint: Solid state-colored circle
                cv2.circle(frame, pt, 4, state_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_person_card(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        track_id: int,
        state: str,
        features: Dict[str, float],
        occ_stats: Dict[str, float],
        state_info: Dict[str, Any]
    ):
        angle = features.get("torso_angle", 90.0)
        v_y = features.get("vertical_velocity", 0.0)
        card_w, card_h = 220, 85
        card_x = max(10, x1)
        card_y = max(45, y1 - card_h - 10)

        # Draw semi-transparent card background
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 38), -1)
        state_color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + 5, card_y + card_h), state_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Header: ID & State
        cv2.putText(frame, f"PERSON #{track_id} | {state}", (card_x + 12, card_y + 18),
                    self.font, 0.42, state_color, 1, cv2.LINE_AA)

        # Torso Angle with bar gauge
        angle_bar = int((angle / 90.0) * 50)
        cv2.putText(frame, f"Torso Angle: {angle:.1f} deg", (card_x + 12, card_y + 36),
                    self.font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Vertical Velocity
        v_sign = "+" if v_y >= 0 else ""
        v_color = self.COLOR_DANGER if v_y > 1.8 else (200, 200, 200)
        cv2.putText(frame, f"V_descent: {v_sign}{v_y:.2f} h/s", (card_x + 12, card_y + 54),
                    self.font, 0.38, v_color, 1, cv2.LINE_AA)

        # Occlusion Status
        if occ_stats.get("is_partially_occluded", False):
            occ_text = f"Occlusion: {int((1.0 - occ_stats['total_visibility'])*100)}% (IMPUTED)"
            cv2.putText(frame, occ_text, (card_x + 12, card_y + 72),
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
