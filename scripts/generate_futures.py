"""Generate REAL future rollouts with Stable Video Diffusion (GPU).

From a seed frame, sample N imagined futures (different seeds), score each for
danger with few-shot CLIP anchors, and write:
  - rollout_0.mp4 .. rollout_{N-1}.mp4   (drop-in sources for the dashboard's
    Decoded Futures tiles)
  - futures.json  [{rollout, severity, danger}]

The dashboard can then point its 3 future <video> tags at rollout_*.mp4 and read
danger from futures.json (same pattern as logs.json) — swapping the fake
rollouts for genuine generative ones.

Run on the GPU box (CUDA + fp16):
    uv run --extra generative python scripts/generate_futures.py \
        --video gripper.mp4 --seek 30 --rollouts 3 --anchors anchors/

Needs the `generative` extra and the SVD weights (~first run downloads them).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def seed_frame(video, seek_s):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_s * 1000.0)
    ok, f = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"could not read {video} @ {seek_s}s")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="gripper.mp4", help="seed video")
    ap.add_argument("--seek", type=float, default=30.0, help="seed timestamp (s)")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--num-frames", type=int, default=14)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--anchors", default="anchors")
    ap.add_argument("--svd", default="stabilityai/stable-video-diffusion-img2vid-xt")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("Stable Video Diffusion requires a CUDA GPU.")

    from src.future_preview import AnchorDangerScorer  # reuse CLIP anchor scorer

    print("loading SVD (first run downloads weights)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(args.svd, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda"); pipe.enable_model_cpu_offload()
    scorer = AnchorDangerScorer(args.anchors)
    try:
        scorer.load()
    except Exception as e:
        print(f"[anchors] disabled ({e})")

    seed = Image.fromarray(seed_frame(args.video, args.seek)).resize((1024, 576))
    manifest = []
    for i in range(args.rollouts):
        gen = torch.Generator(device="cuda").manual_seed(1000 + i)  # distinct future per seed
        print(f"rollout {i}: sampling future...")
        frames = pipe(seed, decode_chunk_size=8, num_frames=args.num_frames, generator=gen).frames[0]
        rgb = [np.asarray(f) for f in frames]
        danger = scorer.score_frames(rgb) if scorer.available else 0.0
        sev = "critical" if danger > args.threshold else "warning" if danger > args.threshold * 0.5 else "info"

        h, w = rgb[0].shape[:2]
        out = f"rollout_{i}.mp4"
        wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for fr in rgb:
            wr.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        wr.release()
        manifest.append({"rollout": i, "severity": sev, "danger": round(float(danger), 3), "src": out})
        print(f"  -> {out}  danger={danger:.3f}  {sev}")

    with open("futures.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote futures.json + {args.rollouts} rollout clips. "
          f"Point the dashboard's .fut <video> at rollout_*.mp4 to use them.")


if __name__ == "__main__":
    main()
