"""
========================================================================================
  AI PERSON & ELDERLY FALL DETECTION SYSTEM (INTERACTIVE RUNNER)
  Features:
    [1] Live Webcam
    [2] Custom Video File (Auto-lists MP4s in folder or accepts Drag & Drop)
    [3] Synthetic Occlusion Fall Demo
    [4] Exit
========================================================================================
"""

import os
import sys
import glob
import argparse
from typing import Optional, List, Dict, Any
import cv2
from fall_detection_all_in_one import run_pipeline, generate_fall_test_video


def get_user_choice() -> str:
    print("\n" + "=" * 70)
    print("  🛡️  AI PERSON & ELDERLY FALL DETECTION SYSTEM")
    print("  Advanced Occlusion Resilience & Kinematic Tracking")
    print("=" * 70)
    print("  Select an input option:")
    print("   [1] 📹 Live Webcam (Real-time monitoring)")
    print("   [2] 📁 Custom Video File (.mp4, .avi, .mkv, .mov)")
    print("   [3] 🧪 Synthetic Occlusion Fall Demo (Built-in test)")
    print("   [4] ❌ Exit")
    print("=" * 70)

    while True:
        try:
            choice = input("Enter your choice (1/2/3/4) [default: 1]: ").strip()
            if not choice:
                choice = "1"
            if choice in ["1", "2", "3", "4"]:
                return choice
            print("⚠️ Invalid choice. Please enter 1, 2, 3, or 4.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def select_custom_video() -> Optional[str]:
    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    all_videos = []
    
    # Search root and sample_videos directory
    for ext in extensions:
        all_videos.extend(glob.glob(ext))
        all_videos.extend(glob.glob(os.path.join("sample_videos", ext)))

    # Filter out generated outputs and test artifacts
    source_videos = [
        v for v in all_videos
        if not os.path.basename(v).startswith("output_")
        and not os.path.basename(v).startswith("test")
        and not os.path.basename(v).startswith("live_")
        and v != "demo_fall_with_occlusion.mp4"
        and v != "fall_detection_output.mp4"
        and os.path.getsize(v) > 1024
    ]

    print("\n" + "-" * 70)
    print("  📁 SELECT A VIDEO TO ANALYZE")
    print("-" * 70)

    if source_videos:
        print("  Available Benchmark & Test Videos:")
        for idx, vid in enumerate(source_videos, 1):
            file_size_kb = os.path.getsize(vid) // 1024
            tag = "🌟 [BENCHMARK]" if "sample_videos" in vid else "📹 [LOCAL]"
            print(f"   [{idx}] {tag} {vid} ({file_size_kb} KB)")
        print(f"   [0] Enter a different file path / Drag & Drop any external video")
        print("   [b] Back to Main Menu")
        print("-" * 70)

        while True:
            choice = input(f"Select video number (1-{len(source_videos)}) or paste path: ").strip()
            if choice.lower() == 'b':
                return None
            
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(source_videos):
                    return source_videos[num - 1]
                elif num == 0:
                    break
            
            clean_path = choice.strip('"').strip("'")
            if os.path.exists(clean_path):
                return clean_path
            
            print("⚠️ Invalid choice or file not found. Please try again.")

    while True:
        path_input = input("\nEnter video path (or drag & drop file here, 'b' for back): ").strip()
        if path_input.lower() == 'b':
            return None
        
        clean_path = path_input.strip('"').strip("'")
        if os.path.exists(clean_path):
            if os.path.getsize(clean_path) < 1024:
                print(f"⚠️ Warning: File '{clean_path}' is empty or corrupt. Please select a valid video.")
                continue
            return clean_path
        else:
            print(f"❌ File not found: '{clean_path}'. Please check the path and try again.")


def main():
    parser = argparse.ArgumentParser(description="AI Fall Detection Interactive Runner")
    parser.add_argument("--source", type=str, default=None, help="Direct source bypass ('0', 'video.mp4', 'demo')")
    args = parser.parse_args()

    if args.source is not None:
        run_pipeline(
            source=args.source,
            save_path="fall_detection_output.mp4",
            show=True,
            conf=0.35
        )
        return

    while True:
        choice = get_user_choice()

        if choice == "1":
            cam_idx = input("Enter Webcam Index [default: 0]: ").strip()
            source = cam_idx if cam_idx else "0"
            save_file = "webcam_record.mp4"
            show_window = True

        elif choice == "2":
            source = select_custom_video()
            if source is None:
                continue
            base_name = os.path.splitext(os.path.basename(source))[0]
            save_file = f"output_{base_name}.mp4"
            show_window = True

        elif choice == "3":
            demo_file = "demo_fall_with_occlusion.mp4"
            if not os.path.exists(demo_file):
                print("\n[Demo] Generating synthetic fall & occlusion scenario...")
                generate_fall_test_video(demo_file)
            source = demo_file
            save_file = "fall_detection_output.mp4"
            show_window = True

        elif choice == "4":
            print("\n👋 Exiting AI Fall Detection System. Goodbye!")
            break

        # Run pipeline
        print(f"\n🚀 Launching Fall Detection Pipeline on: {source}...")
        print("💡 [Tip] Press 'q' or 'ESC' in the video window anytime to stop and return to menu.\n")
        
        try:
            run_pipeline(
                source=source,
                save_path=save_file,
                show=show_window,
                conf=0.35
            )
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")

        print("\n" + "-" * 70)
        input("✅ Analysis finished! Press [Enter] to return to the Main Menu...")


if __name__ == "__main__":
    main()
