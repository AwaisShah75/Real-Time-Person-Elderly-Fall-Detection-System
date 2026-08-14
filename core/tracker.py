"""
Multi-Person Tracklet Manager
Tracks identities across frames and maintains sliding-window kinematic buffers.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class Tracklet:
    """
    Maintains temporal state, landmark history, and state machine for a single person.
    """
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

    def mark_missed(self):
        self.missing_frames += 1


class SimpleTracker:
    """
    Fast IoU and Centroid-based tracker to maintain tracklet continuity.
    """
    def __init__(self, iou_thresh: float = 0.3, max_missing: int = 20):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.tracklets: Dict[int, Tracklet] = {}
        self.next_id: int = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Tracklet]:
        """
        Matches detections to existing tracklets or creates new ones.
        """
        det_bboxes = [d["bbox"] for d in detections]
        det_kps = [d["keypoints"] for d in detections]

        matched_tracks = set()
        matched_dets = set()

        # Match existing tracklets to detections based on IoU and centroid distance
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
                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)
                # Imputed kps initially matches raw kps until processed by occlusion engine
                track.update(det_bboxes[best_det_idx], det_kps[best_det_idx], det_kps[best_det_idx])
            else:
                track.mark_missed()

        # Remove dead tracklets
        for track_id, track in list(self.tracklets.items()):
            if track.missing_frames > self.max_missing:
                del self.tracklets[track_id]

        # Create new tracklets for unmatched detections
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

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)
