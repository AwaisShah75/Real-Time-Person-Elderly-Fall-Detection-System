"""
Multi-Stage Fall State Machine
Manages transition states (Monitoring -> Rapid Descent -> Impact -> Inactivity Check -> Pre-Alert -> Confirmed Fall)
to ensure near-zero false alarms.
"""

from typing import Dict, Any, Tuple
from core.tracker import Tracklet


class FallStateMachine:
    """
    Real-Time Fall State Machine with instant-trigger alerts on falling motion and impact.
    """

    STATE_MONITORING = "MONITORING"
    STATE_FALL_DETECTED = "FALL_DETECTED"
    STATE_IMPACT = "IMPACT"
    STATE_INACTIVITY_CHECK = "INACTIVITY_CHECK"
    STATE_CONFIRMED_FALL = "CONFIRMED_FALL"

    def __init__(
        self,
        fps: float = 30.0,
        fall_velocity_thresh: float = 1.2,
        flat_angle_thresh: float = 55.0,
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
            if angle < self.flat_angle_thresh or aspect_ratio > 0.85 or in_occ_zone:
                if tracklet.state_timer >= self.inactivity_sec:
                    next_state = self.STATE_CONFIRMED_FALL
                    event_trigger = "[EMERGENCY] CONFIRMED EMERGENCY FALL ON GROUND"
                else:
                    next_state = self.STATE_FALL_DETECTED
            elif angle > self.standing_angle_thresh and v_y < -0.3:
                next_state = self.STATE_MONITORING
                event_trigger = "Subject stood upright -> Reset"

        elif current == self.STATE_CONFIRMED_FALL:
            if angle > self.standing_angle_thresh and stillness > 20.0:
                next_state = self.STATE_MONITORING
                event_trigger = "Subject upright and moving -> Reset"

        elif current == self.STATE_MONITORING:
            pass

        tracklet.state = next_state

        return next_state, {
            "state_timer": tracklet.state_timer,
            "countdown": max(0.0, self.inactivity_sec - tracklet.state_timer),
            "event_trigger": event_trigger
        }
