"""Offline 'imagine the future' demo artifact (GPU).

Takes an image (or the first frame of a video), generates a short predicted
future clip with Stable Video Diffusion, scores each predicted frame for danger
with few-shot CLIP anchors, and writes an annotated MP4: the imagined future
playing with a live danger bar + VETO banner. This is the reliable, shareable
"wow" for a hackathon — runs once on the GPU box, produces a clip you can show.

Usage (on the GPU machine):
    uv run --extra generative python scripts/imagine_future.py \
        --image scene.jpg --anchors anchors/ --out imagined_future.mp4

`anchors/` should contain example frames under anchors/dangerous and anchors/safe.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="seed image; or use --video")
    ap.add_argument("--video", help="seed from the first frame of this video")
    ap.add_argument("--anchors", default="anchors", help="few-shot anchors dir (dangerous/ + safe/)")
    ap.add_argument("--out", default="imagined_future.mp4")
    ap.add_argument("--num-frames", type=int, default=14)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--svd", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    # seed frame (RGB)
    if args.image:
        seed = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        ok, f = cap.read(); cap.release()
        if not ok:
            raise SystemExit("could not read video")
        seed = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    else:
        raise SystemExit("pass --image or --video")

    from src.future_preview import AnchorDangerScorer, FutureFramePredictor

    print("loading models (first run downloads weights)...")
    predictor = FutureFramePredictor(args.svd, num_frames=args.num_frames)
    scorer = AnchorDangerScorer(args.anchors)
    predictor.load()
    scorer.load()

    print("imagining the future...")
    future = predictor.predict(seed)  # list of RGB frames

    # per-frame danger (cumulative max so the bar ratchets up as danger appears)
    dangers = [scorer.score_frames([f]) for f in future] if scorer.available else [0.0] * len(future)
    running = np.maximum.accumulate(np.array(dangers))

    h, w = future[0].shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for frame_rgb, d in zip(future, running):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        danger = bool(d > args.threshold)
        col = (0, 0, 255) if danger else (0, 200, 0)
        label = "VETO: predicted danger" if danger else "predicted: safe"
        cv2.rectangle(bgr, (0, 0), (w, 30), col, -1)
        cv2.putText(bgr, f"IMAGINED FUTURE  |  {label}  d={d:.2f}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frac = float(np.clip((d + 0.2) / 0.5, 0, 1))
        cv2.rectangle(bgr, (10, h - 20), (10 + int(frac * (w - 20)), h - 12), col, -1)
        writer.write(bgr)
    writer.release()
    print(f"wrote {args.out}  (peak danger {running.max():.2f}, "
          f"{'VETO' if running.max() > args.threshold else 'safe'})")


if __name__ == "__main__":
    main()
