# 🛡️ Real-Time Person & Elderly Fall Detection System
### Advanced Occlusion Resilience, Kinematic State Machine & Edge AI Architecture

An enterprise-grade, edge-compatible AI computer vision system designed to detect accidental falls in real-time under severe visual occlusions (behind furniture, tables, beds, couches) while completely eliminating false positives from daily activities such as sitting down, tying shoelaces, or sleeping in bed.

---

## 🌟 Key Highlights

- **⚡ Instant-Trigger Fall Detection**: Detects human falls the millisecond a rapid descent ($v_y > 0.7\text{ h/s}$) or posture collapse ($\Delta \theta / \Delta t > 35^\circ/\text{s}$) occurs, triggering immediate visual and dispatch alerts.
- **🛏️ Smart Resting / Sleeping False-Alarm Rejection**: Biomechanically differentiates between an accidental fall and peaceful horizontal resting/sleeping on a bed (`MONITORING (RESTING)` in calm green, zero false sirens).
- **👁️ 4-Tier Occlusion Resilience**:
  1. **Kinematic Bone Constraint Imputation**: Mathematically reconstructs blocked knees and ankles using anatomical anthropometric ratios.
  2. **Temporal Velocity Extrapolation**: Carries momentum vectors through visual blockages.
  3. **Furniture ROI Tracking**: Monitors trajectories descending behind known occlusion barriers (coffee tables, beds, sofas).
  4. **Proximity & Bone Length Clamping**: Enforces strict skeletal bounds to eliminate spurious spiderweb lines.
- **🚀 100% Edge CPU Ready**: Powered by a lightweight pose estimation engine (~6.5 MB). Runs at full speed on standard CPUs without requiring a dedicated GPU or custom model training.
- **🔄 Continuous Interactive Runner**: One-click launcher with auto-looping menu, video auto-discovery, and drag-and-drop support.

---

## 🚀 Quickstart Guide (1 Command to Run)

You only need **ONE** command:

```bash
python run.py
```
*(Or on Windows, simply double-click **`run.bat`**)*

### 🎮 Interactive Menu Options:

```text
======================================================================
  🛡️  AI PERSON & ELDERLY FALL DETECTION SYSTEM
  Advanced Occlusion Resilience & Kinematic Tracking
======================================================================
  Select an input option:
   [1] 📹 Live Webcam (Real-time monitoring)
   [2] 📁 Custom Video File (Auto-detects videos + Drag & Drop)
   [3] 🧪 Synthetic Occlusion Fall Demo (Built-in test)
   [4] ❌ Exit
======================================================================
```

- **Option [1] (Webcam)**: Streams live camera feed with real-time skeleton tracking and HUD telemetry.
- **Option [2] (Custom Video)**: Automatically lists all available test videos and benchmark clips with one-key selection (`1`, `2`, `3`, `4`, `5`), or accepts any drag-and-dropped `.mp4` / `.avi` file.
- **Option [3] (Synthetic Demo)**: Generates and executes a synthetic scenario of a person walking and falling behind an occlusion barrier.
- **Option [4] (Exit)**: Safely closes the application.

---

## 📦 Included Benchmark Video Suite (`sample_videos/`)

The repository includes real-world benchmark fall test videos:

| # | Video File | Scenario & Purpose | Expected Result |
|---|---|---|---|
| **1** | `sample_videos/standing_fall_1.mp4` | 🏃 Sudden collapse fall from standing | Instant `[ALERT] FALL DETECTED` |
| **2** | `sample_videos/standing_fall_2.mp4` | 🍌 Slip & backward fall from standing | Instant `[ALERT] FALL DETECTED` |
| **3** | `sample_videos/sitting_to_fall.mp4` | 🪑 Senior citizen falling off chair | Instant `[ALERT] FALL DETECTED` |
| **4** | `sample_videos/bed_rollout_fall.mp4` | 🛏️ Roll-out fall from bed onto floor | Instant `[ALERT] FALL DETECTED` |
| **5** | `sample_videos/normal_activity_no_fall.mp4` | 🚶 Walking, sitting & sleeping (ADL) | `MONITORING (RESTING)` (Zero False Alarms) |

---

## 🧠 Technical Architecture & Biomechanical Pipeline

```
  ┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ Video / Webcam  │ ──> │ YOLOv8-Pose Backbone │ ──> │  Occlusion Resilience  │
  │     Stream      │     │  Keypoint Extractor  │     │   & Bone Imputation    │
  └─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                    │
  ┌─────────────────┐     ┌──────────────────────┐                  ▼
  │ HUD Visualizer  │ <── │ Multi-Stage Fall FSM │ <── ┌────────────────────────┐
  │  & Alert Engine │     │  State Transitions   │     │  Kinematic Extractor   │
  └─────────────────┘     └──────────────────────┘     │ (v_y, Angle, ω, Energy)│
                                                       └────────────────────────┘
```

### 1. Biomechanical Features
- **Torso Angle ($\theta_{torso}$)**: Angle formed by the midpoint of shoulders to midpoint of hips relative to the floor horizontal ($90^\circ = \text{standing}, 0^\circ = \text{flat on ground}$).
- **Vertical Descent Rate ($v_y$)**: Normalized downward velocity measured in subject bounding box heights per second ($\text{h/s}$).
- **Angular Collapse Rate ($\omega = \frac{d\theta}{dt}$)**: Rate of postural angle drop in degrees per second.
- **Stillness Energy ($E_{motion}$)**: Temporal variance across visible anatomical landmarks to confirm post-impact immobility.

### 2. State Machine Logic Flow
```
 [ MONITORING ] ──( v_y > 0.7 h/s OR ω > 35°/s )──> [ FALL_DETECTED ]
       ▲                                                     │
       │                                            ( > 1.5s Stillness )
  ( Stands Up )                                              │
       │                                                     ▼
 [ MONITORING ] <──────( Upright Motion )─────── [ CONFIRMED_FALL ]
```

---

## 📁 Repository File Structure

```
d:\A New Fall Detecion Sytem\
├── run.bat                          # One-click Windows desktop launcher
├── run.py                           # Interactive CLI runner with continuous loop
├── fall_detection_all_in_one.py     # Standalone all-in-one pipeline (Zero sub-dependencies)
├── core/
│   ├── detector.py                  # YOLOv8-Pose wrapper & keypoint extractor
│   ├── tracker.py                   # Multi-person temporal tracklet manager
│   ├── occlusion.py                 # Occlusion engine, Kalman extrapolation & bone imputation
│   ├── kinematics.py                # Biomechanical kinematic feature engine
│   ├── state_machine.py             # Multi-stage Fall State Machine
│   └── visualizer.py                # Telemetry HUD visualizer & alert banners
├── sample_videos/                   # Benchmark fall test dataset
│   ├── standing_fall_1.mp4          # Sudden collapse fall
│   ├── standing_fall_2.mp4          # Slip and fall
│   ├── sitting_to_fall.mp4          # Chair fall
│   ├── bed_rollout_fall.mp4         # Bed roll-out fall
│   └── normal_activity_no_fall.mp4  # Normal ADL baseline (zero false alarm test)
├── test_generator.py                # Synthetic occlusion test scenario generator
├── requirements.txt                 # Clean dependency manifest
├── FALL_DETECTION_SYSTEM_DESIGN.md  # Comprehensive research & mathematical design spec
└── README.md                        # Documentation & quickstart guide
```

---

## 🛠️ Advanced CLI Usage (Direct Bypass)

If you prefer direct command-line execution without the interactive menu:

```bash
# Run on webcam (device 0)
python fall_detection_all_in_one.py --source 0 --show

# Run on a specific video file
python fall_detection_all_in_one.py --source sample_videos/standing_fall_1.mp4 --show --save output_result.mp4

# Run synthetic occlusion demo
python fall_detection_all_in_one.py --source demo --show
```

---

## 📜 Scientific Reference & Design Document
For full mathematical formulations, biomechanical formulas, Kalman filter state equations, and multi-view homography matrices, refer to:
👉 **[FALL_DETECTION_SYSTEM_DESIGN.md](file:///d:/A%20New%20Fall%20Detecion%20Sytem/FALL_DETECTION_SYSTEM_DESIGN.md)**
