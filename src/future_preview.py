"""Generative future-preview track (GPU) — "imagine the near future, veto danger".

Camera-only, automatic continuation: from the current frame, a pretrained video
world model (Stable Video Diffusion, image->video) generates a short plausible
FUTURE clip. We then score that imagined future for danger using few-shot
anchors (CLIP similarity to example "dangerous" vs "safe" frames) and raise a
preventive VETO before anything happens. This is the generative half of the
hybrid demo; the V-JEPA latent track (src/world_model.py) is the latent half.

⚠️ GPU-only. Stable Video Diffusion needs CUDA + fp16 and is heavy (a few
seconds per clip even on GPU). Runs ASYNC as a periodic "glance into the future"
— it never blocks the fast tracks. Untested on macOS-Intel CPU by design.

Heavy deps (diffusers, transformers CLIP, accelerate) live in the `generative`
extra and are imported lazily, so the core app runs without them.
"""
from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass, field

import numpy as np


# ----- danger scoring via few-shot anchors (CLIP) ---------------------------
class AnchorDangerScorer:
    """Score a frame by CLIP-embedding similarity to example dangerous vs safe
    frames. Put example images under <anchors_dir>/dangerous and /safe."""

    def __init__(self, anchors_dir: str, clip_model: str = "openai/clip-vit-base-patch32"):
        self.anchors_dir = anchors_dir
        self.clip_model = clip_model
        self._model = None
        self._proc = None
        self._danger_emb = None   # (Nd, D)
        self._safe_emb = None     # (Ns, D)

    @property
    def available(self) -> bool:
        return self._danger_emb is not None and len(self._danger_emb) > 0

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.clip_model)
        self._proc = CLIPProcessor.from_pretrained(self.clip_model)
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()
        self._danger_emb = self._embed_dir(os.path.join(self.anchors_dir, "dangerous"))
        self._safe_emb = self._embed_dir(os.path.join(self.anchors_dir, "safe"))

    def _embed_dir(self, d: str):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += glob.glob(os.path.join(d, ext))
        if not paths:
            return np.zeros((0, 512), dtype=np.float32)
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        return self._embed_images(imgs)

    def _embed_images(self, imgs):
        import torch

        inp = self._proc(images=imgs, return_tensors="pt")
        if next(self._model.parameters()).is_cuda:
            inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inp)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def score_frames(self, frames_rgb: list[np.ndarray]) -> float:
        """Danger in [~-1, 1]: mean over predicted frames of
        (max sim to a dangerous anchor) - (max sim to a safe anchor)."""
        if not self.available:
            return 0.0
        from PIL import Image

        emb = self._embed_images([Image.fromarray(f) for f in frames_rgb])
        dsim = emb @ self._danger_emb.T  # (F, Nd)
        d = dsim.max(axis=1)
        if self._safe_emb is not None and len(self._safe_emb):
            ssim = emb @ self._safe_emb.T
            d = d - ssim.max(axis=1)
        return float(d.mean())


# ----- generative future predictor (Stable Video Diffusion) -----------------
class FutureFramePredictor:
    """Image->video: predict a short future clip from the current frame."""

    def __init__(self, model_id: str, num_frames: int = 14):
        self.model_id = model_id
        self.num_frames = num_frames
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableVideoDiffusionPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Future-preview (SVD) requires a CUDA GPU.")
        self._pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, variant="fp16"
        )
        self._pipe.to("cuda")
        self._pipe.enable_model_cpu_offload()

    def predict(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        from PIL import Image

        self.load()
        img = Image.fromarray(frame_rgb).resize((1024, 576))
        result = self._pipe(img, decode_chunk_size=8, num_frames=self.num_frames)
        return [np.asarray(f) for f in result.frames[0]]


# ----- async monitor wiring it together -------------------------------------
@dataclass
class FutureState:
    status: str = "idle"            # idle | loading | ready
    danger: float = 0.0             # anchor danger score
    is_danger: bool = False
    future_frames: list = field(default_factory=list)  # predicted RGB frames (thumbnails)


class FuturePreviewMonitor:
    def __init__(self, predictor: FutureFramePredictor, scorer: AnchorDangerScorer | None,
                 interval_s: float, danger_threshold: float):
        self.predictor = predictor
        self.scorer = scorer
        self.interval_s = interval_s
        self.danger_threshold = danger_threshold
        self._latest_frame = None
        self._busy = False
        self._loaded = False
        self._last_t = 0.0
        self._lock = threading.Lock()
        self.state = FutureState()

    def push_frame(self, frame_bgr: np.ndarray):
        import cv2

        self._latest_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def maybe_predict(self, now: float):
        if self._busy or self._latest_frame is None or now - self._last_t < self.interval_s:
            return
        self._last_t = now
        self._busy = True
        frame = self._latest_frame.copy()
        threading.Thread(target=self._work, args=(frame,), daemon=True).start()

    def _work(self, frame: np.ndarray):
        try:
            if not self._loaded:
                with self._lock:
                    self.state.status = "loading"
                self.predictor.load()
                if self.scorer is not None:
                    self.scorer.load()
                self._loaded = True
            frames = self.predictor.predict(frame)
            danger = self.scorer.score_frames(frames) if self.scorer else 0.0
            thumbs = self._thumbs(frames)
            with self._lock:
                self.state.status = "ready"
                self.state.future_frames = thumbs
                self.state.danger = danger
                self.state.is_danger = danger > self.danger_threshold
        except Exception as e:
            print(f"[future_preview] error: {e}")
        finally:
            self._busy = False

    @staticmethod
    def _thumbs(frames, every: int = 2, w: int = 160):
        import cv2

        out = []
        for f in frames[::every]:
            h = int(f.shape[0] * w / f.shape[1])
            out.append(cv2.cvtColor(cv2.resize(f, (w, h)), cv2.COLOR_RGB2BGR))
        return out

    def snapshot(self) -> FutureState:
        with self._lock:
            return FutureState(
                status=self.state.status,
                danger=self.state.danger,
                is_danger=self.state.is_danger,
                future_frames=list(self.state.future_frames),
            )
