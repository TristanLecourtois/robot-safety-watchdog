"""Entry point. Run the live webcam watchdog.

    python main.py            # live webcam
    python main.py --video f  # run on a recorded clip instead
"""
from __future__ import annotations

import argparse

import config as cfg
from src.watchdog import Watchdog, run_webcam


def main():
    ap = argparse.ArgumentParser(description="Robot safety vision watchdog")
    ap.add_argument("--camera", type=int, default=cfg.CONFIG.camera_index)
    ap.add_argument("--video", type=str, default=None, help="path to a video file to analyze")
    ap.add_argument("--worldmodel", action="store_true",
                    help="enable the V-JEPA 2 latent OOD track (needs the worldmodel extra)")
    ap.add_argument("--future", action="store_true",
                    help="enable the generative future-preview track (GPU; needs the generative extra)")
    args = ap.parse_args()

    if args.worldmodel:
        cfg.CONFIG.enable_world_model = True
    if args.future:
        cfg.CONFIG.enable_future_preview = True

    if args.video:
        _run_video(args.video)
    else:
        cfg.CONFIG.camera_index = args.camera
        run_webcam(cfg.CONFIG)


def _run_video(path: str):
    import time
    import cv2
    from src import overlay

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
        overlay.draw(frame, detections, hands, analysis, rationale, sev,
                     latent=wd.world_state(), future=wd.future_state())
        cv2.imshow("Robot Safety Watchdog", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
