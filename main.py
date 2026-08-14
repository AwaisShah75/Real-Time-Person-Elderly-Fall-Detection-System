"""
Main Pipeline Runner for AI Fall Detection & Occlusion Resilience System
Executes real-time pose estimation, occlusion recovery, kinematic tracking,
and state machine fall verification.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from core.detector import PoseDetector
from core.tracker import SimpleTracker, Tracklet
from core.occlusion import OcclusionEngine
from core.kinematics import KinematicExtractor
from core.state_machine import FallStateMachine
from core.visualizer import Visualizer
from test_generator import generate_fall_test_video


class FallDetectionPipeline:
    """
    Main orchestration class that coordinates all modules.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        device: str = None,
        fps: float = 30.0
    ):
        self.fps = fps
        self.detector = PoseDetector(model_name=model_name, conf_thresh=conf_thresh, device=device)
        self.tracker = SimpleTracker(iou_thresh=0.25, max_missing=20)
        self.occlusion_engine = OcclusionEngine(conf_threshold=0.25)
        self.kinematics = KinematicExtractor(fps=fps)
        self.state_machine = FallStateMachine(fps=fps, inactivity_sec=4.0, pre_alert_sec=5.0)
        self.visualizer = Visualizer()

    def add_furniture_zone(self, name: str, polygon: List[tuple]):
        self.occlusion_engine.add_occlusion_zone(name, polygon)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs full pipeline on a single frame.

        Returns:
            rendered_frame: np.ndarray
            alerts: List of active emergency alerts
        """
        # Step 1: Detect Persons & Keypoints
        detections = self.detector.detect(frame)

        # Step 2: Update Multi-Person Tracker
        tracklets = self.tracker.update(detections)

        tracklet_data: Dict[int, Dict[str, Any]] = {}
        alerts: List[Dict[str, Any]] = []

        for track in tracklets:
            t_id = track.track_id
            curr_bbox = track.bbox_history[-1]
            raw_kps = track.raw_kps_history[-1]

            # Step 3: Occlusion Analysis & Keypoint Imputation
            imputed_kps, occ_mask, occ_stats = self.occlusion_engine.process_keypoints(
                current_kps=raw_kps,
                history_kps=track.imputed_kps_history,
                bbox=curr_bbox
            )
            # Update tracklet imputed history
            track.imputed_kps_history[-1] = imputed_kps

            # Check if person intersects an occlusion zone
            centroid = (
                (curr_bbox[0] + curr_bbox[2]) / 2.0,
                (curr_bbox[1] + curr_bbox[3]) / 2.0
            )
            in_zone, zone_name = self.occlusion_engine.is_in_occlusion_zone(centroid)
            track.in_occlusion_zone = in_zone
            track.occlusion_zone_name = zone_name

            # Step 4: Compute Kinematic Features
            features = self.kinematics.extract_features(
                keypoints_history=track.imputed_kps_history,
                bbox_history=track.bbox_history,
                occlusion_stats=occ_stats
            )

            # Step 5: Advance State Machine
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
                alerts.append({
                    "track_id": t_id,
                    "state": state,
                    "info": state_info,
                    "features": features
                })

        # Step 6: Render HUD Visualizer
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
    print("=" * 70)
    print("  REAL-TIME PERSON & ELDERLY FALL DETECTION SYSTEM")
    print("  With Advanced Occlusion Resilience & Multi-Stage State Machine")
    print("=" * 70)

    if source == "demo":
        demo_file = "demo_fall_with_occlusion.mp4"
        if not os.path.exists(demo_file):
            print("[Pipeline] Generating synthetic test video with occlusion barrier...")
            generate_fall_test_video(demo_file)
        source = demo_file

    is_cam = str(source).isdigit()
    if is_cam:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[Error] Could not open video source: '{source}'. Check file path or camera permissions.")
        return

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

    # Video Writer
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Output will be saved to: {save_path}")

    pipeline = FallDetectionPipeline(conf_thresh=conf, device=device, fps=fps)

    # If demo video, register the table occlusion zone
    if "demo" in str(source):
        pipeline.add_furniture_zone("Coffee Table", [(300, 310), (620, 310), (620, 440), (300, 440)])

    frame_count = 0
    t_start = time.time()
    window_name = "AI Fall Detection & Occlusion Resilience"

    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

            # Process frame
            vis_frame, alerts = pipeline.process_frame(frame)

            t_elapsed = time.time() - t_frame_start
            current_fps = 1.0 / max(t_elapsed, 1e-4)
            pipeline.fps = current_fps

            # Log any active emergency alerts
            for alert in alerts:
                if alert["state"] in ["FALL_DETECTED", "IMPACT", "CONFIRMED_FALL"]:
                    print(f"🚨 [EMERGENCY ALERT] Frame {frame_count}: Fall detected for Person #{alert['track_id']}! "
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
    print("=" * 70)
    print(f"  Execution Complete! Processed {frame_count} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    if save_path:
        print(f"  Processed video saved at: {os.path.abspath(save_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fall Detection & Occlusion System")
    parser.add_argument("--source", type=str, default="demo", help="Path to video file, webcam index (0), or 'demo'")
    parser.add_argument("--save", type=str, default="fall_detection_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", default=False, help="Display live OpenCV window")
    parser.add_argument("--conf", type=float, default=0.35, help="Pose detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or None for auto")
    
    args = parser.parse_args()
    run_pipeline(source=args.source, save_path=args.save, show=args.show, conf=args.conf, device=args.device)
