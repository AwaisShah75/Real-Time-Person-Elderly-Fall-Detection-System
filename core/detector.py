"""
Pose Detection Module using YOLOv8-Pose
Extracts 17 COCO Keypoints per detected person in real-time.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO


class PoseDetector:
    """
    Wraps YOLOv8-Pose model for efficient single/multi-person landmark estimation.
    """
    
    COCO_KEYPOINTS = [
        "nose",          # 0
        "left_eye",      # 1
        "right_eye",     # 2
        "left_ear",      # 3
        "right_ear",     # 4
        "left_shoulder", # 5
        "right_shoulder",# 6
        "left_elbow",    # 7
        "right_elbow",   # 8
        "left_wrist",    # 9
        "right_wrist",   # 10
        "left_hip",      # 11
        "right_hip",     # 12
        "left_knee",     # 13
        "right_knee",    # 14
        "left_ankle",    # 15
        "right_ankle"    # 16
    ]

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: Optional[str] = None):
        """
        Args:
            model_name: YOLO pose model path/name (default: lightweight yolov8n-pose.pt)
            conf_thresh: Minimum detection confidence threshold
            device: 'cpu', 'cuda', or None for auto-detection
        """
        self.conf_thresh = conf_thresh
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[PoseDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs pose estimation on a single frame.

        Returns list of dicts:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "det_conf": float,
                "keypoints": np.ndarray of shape (17, 3) where columns are [x, y, confidence]
            },
            ...
        ]
        """
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
        
        # Keypoints data shape: (N, 17, 3) -> [x, y, conf] or (N, 17, 2)
        kps_data = result.keypoints.data.cpu().numpy()

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            det_conf = float(confs[i])
            kps = kps_data[i]  # shape (17, 3) or (17, 2)

            if kps.shape[-1] == 2:
                # If confidence column missing, synthesize full confidence for visible points
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
