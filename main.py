"""Entry point. Run the live watchdog.

    python main.py                      # single webcam (camera 0)
    python main.py --camera 1           # specific camera index
    python main.py --cameras 0 1        # two cameras side by side
    python main.py --cameras 0 1 2      # three cameras in a grid
    python main.py --cameras rtsp://… 0 # mix of streams + webcam
    python main.py --video path.mp4     # single video file
"""
from __future__ import annotations

import argparse

import config as cfg


def _parse_source(s: str) -> int | str:
    """Camera index (int) or video/stream path (str)."""
    try:
        return int(s)
    except ValueError:
        return s


def main():
    ap = argparse.ArgumentParser(description="Robot safety vision watchdog")
    # Single-camera convenience args (backward-compatible).
    ap.add_argument("--camera", type=int, default=None,
                    help="Single camera index (default: 0)")
    ap.add_argument("--video", type=str, default=None,
                    help="Path to a video file to analyze (single-camera)")
    # Multi-camera: one or more sources.
    ap.add_argument("--cameras", nargs="+", type=_parse_source, default=None,
                    metavar="SOURCE",
                    help="One or more camera indices / video paths / stream URLs. "
                         "When more than one source is given, multi-camera mode is used.")
    args = ap.parse_args()

    if args.video:
        _run_video(args.video)
        return

    # Determine sources list.
    sources: list[int | str]
    if args.cameras:
        sources = args.cameras
    elif args.camera is not None:
        sources = [args.camera]
    else:
        sources = [cfg.CONFIG.camera_index]

    if len(sources) == 1:
        cfg.CONFIG.camera_index = sources[0] if isinstance(sources[0], int) else 0
        from src.watchdog import run_webcam
        run_webcam(cfg.CONFIG)
    else:
        cfg.CONFIG.cameras = [
            cfg.CameraConfig(source=src, label=f"cam_{src}")
            for src in sources
        ]
        from src.multi_camera import run_multicam
        run_multicam(cfg.CONFIG)


def _run_video(path: str):
    import time
    import cv2
    from src import overlay
    from src.watchdog import Watchdog

    wd = Watchdog(cfg.CONFIG)
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        detections, hands, analysis = wd.process_frame(frame, now)
        rationale, vlm_sev = wd.verdict_banner()
        sev = analysis.max_severity or vlm_sev
        overlay.draw(frame, detections, hands, analysis, rationale, sev)
        cv2.imshow("Robot Safety Watchdog", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
