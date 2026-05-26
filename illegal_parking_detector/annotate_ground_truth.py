"""Controls:
  SPACE       Pause / Resume
  S           Mark START of an illegal parking violation
  E           Mark END of the current violation
  D           Delete the last annotation
  W           Save annotations to JSON and continue
  Q / ESC     Save annotations and quit

Usage:
  python annotate_ground_truth.py --video ../../cctv_recordings/f1.mp4
  python annotate_ground_truth.py --video ../../cctv_recordings/f1.mp4 --output gt_f1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate illegal parking ground-truth violations in a video."
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input CCTV video file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path. Defaults to <video_stem>_gt.json in same directory.",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (e.g. 0.5 = half speed, 2.0 = double speed).",
    )
    return parser.parse_args()


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.f"""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def draw_hud(frame, current_time: float, fps: float, paused: bool,
             annotations: list, active_start: float | None):
    """Draw the annotation HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Time & status
    time_str = f"Time: {format_time(current_time)}"
    cv2.putText(frame, time_str, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status = "PAUSED" if paused else "PLAYING"
    status_color = (0, 200, 255) if paused else (0, 255, 0)
    cv2.putText(frame, status, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Annotation count
    count_str = f"Annotations: {len(annotations)}"
    cv2.putText(frame, count_str, (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Active recording indicator
    if active_start is not None:
        rec_str = f"REC: violation started @ {format_time(active_start)}"
        cv2.putText(frame, rec_str, (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Blinking red circle
        if int(current_time * 3) % 2 == 0:
            cv2.circle(frame, (280, 55), 8, (0, 0, 255), -1)

    # Controls bar at bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    controls = "SPACE:Pause  S:Start  E:End  D:Delete  W:Save  Q:Quit"
    cv2.putText(frame, controls, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


def save_annotations(annotations: list, output_path: str, video_path: str):
    """Save annotations to JSON."""
    data = {
        "video_source": Path(video_path).name,
        "violations": [
            {
                "start_time_sec": round(a["start"], 3),
                "end_time_sec": round(a["end"], 3),
            }
            for a in annotations
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {len(annotations)} annotation(s) -> {output_path}")


def main():
    args = parse_args()

    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Default output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_gt.json")

    # Load existing annotations if file exists
    annotations = []
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for v in existing.get("violations", []):
            annotations.append({
                "start": v["start_time_sec"],
                "end": v["end_time_sec"],
            })
        print(f"[Info] Loaded {len(annotations)} existing annotation(s) from {output_path}")

    print(f"\n{'='*55}")
    print(f"  Ground-Truth Annotation Tool")
    print(f"  Video:    {Path(video_path).name}")
    print(f"  Duration: {format_time(duration)}  ({total_frames} frames @ {fps:.1f} FPS)")
    print(f"  Output:   {output_path}")
    print(f"  Speed:    {args.speed}x")
    print(f"{'='*55}")
    print(f"  SPACE = Pause/Play")
    print(f"  S     = Mark START of illegal parking")
    print(f"  E     = Mark END of illegal parking")
    print(f"  D     = Delete last annotation")
    print(f"  W     = Save annotations")
    print(f"  Q/ESC = Save & Quit")
    print(f"{'='*55}\n")

    paused = False
    active_start = None
    wait_ms = max(1, int((1000 / fps) / args.speed))

    window_name = "Ground-Truth Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Re-draw current frame when paused
            pass

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_idx / fps

        display = frame.copy()
        display = draw_hud(display, current_time, fps, paused,
                           annotations, active_start)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(wait_ms if not paused else 50) & 0xFF

        if key == ord(" "):
            paused = not paused
            state_str = "PAUSED" if paused else "PLAYING"
            print(f"  [{state_str}] @ {format_time(current_time)}")

        elif key == ord("s") or key == ord("S"):
            if active_start is not None:
                print(f"  [Warning] Violation already started @ {format_time(active_start)}. Press E to end it first.")
            else:
                active_start = current_time
                print(f"  [START] Violation started @ {format_time(current_time)}")

        elif key == ord("e") or key == ord("E"):
            if active_start is None:
                print("  [Warning] No active violation. Press S to start one first.")
            else:
                annotations.append({
                    "start": active_start,
                    "end": current_time,
                })
                print(
                    f"  [END] Violation #{len(annotations)}: "
                    f"{format_time(active_start)} -> {format_time(current_time)} "
                    f"(duration: {current_time - active_start:.1f}s)"
                )
                active_start = None

        elif key == ord("d") or key == ord("D"):
            if annotations:
                removed = annotations.pop()
                print(
                    f"  [DELETE] Removed annotation: "
                    f"{format_time(removed['start'])} -> {format_time(removed['end'])}"
                )
            else:
                print("  [Warning] No annotations to delete.")

        elif key == ord("w") or key == ord("W"):
            save_annotations(annotations, output_path, video_path)

        elif key in (ord("q"), ord("Q"), 27):  # Q or ESC
            # Close active violation if any
            if active_start is not None:
                annotations.append({
                    "start": active_start,
                    "end": current_time,
                })
                print(
                    f"  [AUTO-END] Violation #{len(annotations)}: "
                    f"{format_time(active_start)} -> {format_time(current_time)}"
                )
                active_start = None
            break

    # Final save
    save_annotations(annotations, output_path, video_path)

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Annotation Summary")
    print(f"{'='*55}")
    for i, a in enumerate(annotations):
        dur = a["end"] - a["start"]
        print(f"  #{i+1}  {format_time(a['start'])} -> {format_time(a['end'])}  ({dur:.1f}s)")
    print(f"{'='*55}")
    print(f"  Total: {len(annotations)} violation(s)")
    print(f"  Saved: {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
