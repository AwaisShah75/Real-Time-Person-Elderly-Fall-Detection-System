"""
Synthetic Video Generator for Fall & Occlusion Testing
Generates realistic human motion sequences with furniture occlusion barriers
to verify the fall detection pipeline without requiring a physical camera.
"""

import cv2
import numpy as np


def generate_fall_test_video(output_path: str = "demo_fall_with_occlusion.mp4", duration_sec: int = 14, fps: int = 30):
    """
    Renders a realistic human silhouette and skeleton executing:
      - 0s - 3s: Walking upright towards an occlusion barrier (coffee table).
      - 3s - 5s: Lower body passes behind the barrier (occlusion begins).
      - 5s - 6s: Accidental slip & rapid downward descent fall!
      - 6s - 14s: Motionless horizontal posture on the ground behind barrier (immobility).
    """
    width, height = 854, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Room colors
    bg_color = (240, 240, 245)      # Light room wall
    floor_color = (210, 190, 170)   # Floor tile color
    table_color = (60, 90, 140)     # Dark brown table / furniture

    # Furniture / Occlusion Barrier Coordinates
    table_x1, table_y1 = 300, 310
    table_x2, table_y2 = 620, 440

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Draw Floor Line
        cv2.rectangle(frame, (0, 360), (width, height), floor_color, -1)
        cv2.line(frame, (0, 360), (width, 360), (180, 160, 140), 2)

        # Compute person position & posture based on time phase
        if t < 5.0:
            # Phase 1: Walking upright
            px = int(120 + t * 65)  # walks from x=120 to x=445
            head_y = 160
            torso_angle = 90.0
            is_fallen = False
            leg_swing = np.sin(t * 8) * 20
        elif t < 6.0:
            # Phase 2: Slipping & rapid downward fall
            fall_prog = (t - 5.0) / 1.0
            px = int(445 + fall_prog * 30)
            head_y = int(160 + fall_prog * 180)  # drops from y=160 to y=340
            torso_angle = 90.0 - fall_prog * 75.0  # tilts from 90° to 15°
            is_fallen = True
            leg_swing = 0
        else:
            # Phase 3: Motionless on the floor
            px = 475
            head_y = 345
            torso_angle = 12.0
            is_fallen = True
            leg_swing = 0

        # Draw Human Figure (Head, Torso, Limbs)
        skin_color = (180, 190, 220)
        cloth_color = (120, 60, 40)
        pants_color = (50, 50, 80)

        rad = np.radians(torso_angle)
        torso_len = 90

        # Head & Torso coords
        hx, hy = px, head_y
        hip_x = int(hx - torso_len * np.cos(rad))
        hip_y = int(hy + torso_len * np.sin(rad))

        # Knees & Feet coords
        knee_x = int(hip_x - 45 * np.cos(rad) + leg_swing * 0.3)
        knee_y = int(hip_y + 45 * np.sin(rad))
        foot_x = int(knee_x - 40 * np.cos(rad) - leg_swing * 0.5)
        foot_y = min(height - 10, int(knee_y + 40 * np.sin(rad)))

        # Draw Person Limbs (Before Table is drawn)
        cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), pants_color, 14)
        cv2.line(frame, (knee_x, knee_y), (foot_x, foot_y), pants_color, 12)
        cv2.line(frame, (hx, hy), (hip_x, hip_y), cloth_color, 18)
        cv2.circle(frame, (hx, hy), 16, skin_color, -1)

        # Arms
        cv2.line(frame, (hx, hy + 20), (int(hx + 25 * np.cos(rad)), int(hy + 20 + 35 * np.sin(rad))), cloth_color, 8)

        # Draw Furniture Occlusion Barrier (Table) ON TOP of the person's lower body!
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), table_color, -1)
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), (40, 60, 100), 3)
        
        # Table label
        cv2.putText(frame, "[OCCLUSION BARRIER: TABLE]", (table_x1 + 35, table_y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Room annotations
        status_text = "Phase: Walking" if t < 5.0 else ("Phase: SLIPPING / FALLING" if t < 6.0 else "Phase: POST-FALL IMMOBILITY")
        cv2.putText(frame, f"TEST SCENARIO: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[TestGenerator] Synthetic fall test video successfully created at: {output_path}")


if __name__ == "__main__":
    generate_fall_test_video()
