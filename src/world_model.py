"""World-model track — V-JEPA 2 latent OOD danger monitor.

The symbolic track (YOLO/SAM + rules) and the VLM say *what* is dangerous. This
track adds a holistic, self-supervised signal: a video world model (V-JEPA 2)
encodes short clips into a latent space, we learn what *normal/safe* operation
looks like there, and flag moments that drift out of that distribution as
potentially dangerous — the "the model knows what normal looks like; danger is
surprise" idea. No danger labels needed (out-of-distribution detection).

CPU reality: V-JEPA 2 ViT-L is heavy (~10-20s per clip on an Intel-Mac CPU, plus
a one-time ~80s load). So this runs fully ASYNC in a worker thread and updates a
latent verdict every ~10-20s; it never blocks the fast loop. Treat it as a
periodic "deep glance", not a per-frame detector.

Design:
  VJEPAEncoder      — lazy-loads the model, encodes a list of frames -> 1024-d vec
  LatentOODMonitor  — rolling frame buffer + background encode + OOD scoring +
                      2D PCA map of the latent space (for the "wow" overlay)

The encoder is injected, so the OOD logic is testable with a mock encoder.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ----- the heavy encoder (optional import) ----------------------------------
class VJEPAEncoder:
    """Wraps V-JEPA 2 from HuggingFace transformers. Loaded lazily (the load
    itself is ~80s on CPU), so construct then call `load()` from a worker."""

    def __init__(self, model_id: str, input_size: int = 256):
        self.model_id = model_id
        self.input_size = input_size
        self._model = None
        self._proc = None

    def load(self):
        if self._model is not None:
            return
        import torch
        # transformers' fast video processor calls torch.compiler.is_compiling()
        # which doesn't exist on torch 2.2.x (the macOS-Intel ceiling). Shim it.
        if not hasattr(torch.compiler, "is_compiling"):
            torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
        from transformers import AutoVideoProcessor, VJEPA2Model

        self._proc = AutoVideoProcessor.from_pretrained(self.model_id)
        self._model = VJEPA2Model.from_pretrained(self.model_id)
        self._model.eval()

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames: list of HxWx3 uint8 RGB. Returns a 1D latent vector."""
        import torch

        self.load()
        inp = self._proc(frames, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_vision_features(**inp)  # (1, tokens, dim)
        emb = feats.mean(dim=1).squeeze(0).cpu().numpy()    # mean-pool tokens
        return emb.astype(np.float32)


# ----- OOD state ------------------------------------------------------------
@dataclass
class LatentState:
    status: str = "idle"        # idle | loading | calibrating | monitoring
    progress: str = ""          # e.g. "5/12" during calibration
    danger_score: float = 0.0   # z-score vs the normal baseline
    is_anomaly: bool = False
    # 2D PCA projection for the overlay map.
    normal_2d: np.ndarray | None = None      # (k, 2) baseline cloud
    recent_2d: list[tuple[float, float]] = field(default_factory=list)


class LatentOODMonitor:
    """Buffers frames, encodes clips in the background, learns the normal
    latent distribution, and scores new clips by how far they drift out of it."""

    def __init__(self, encoder, clip_frames: int, calib_clips: int,
                 z_threshold: float, input_size: int = 256):
        self.encoder = encoder
        self.clip_frames = clip_frames
        self.calib_clips = calib_clips
        self.z_threshold = z_threshold
        self.input_size = input_size

        self._buffer: deque[np.ndarray] = deque(maxlen=clip_frames)
        self._normal: list[np.ndarray] = []     # baseline embeddings
        self._mean: np.ndarray | None = None
        self._dist_mu = 0.0
        self._dist_sigma = 1.0
        self._pca = None
        self._busy = False
        self._loaded = False
        self._lock = threading.Lock()
        self.state = LatentState()

    # called from the fast loop every frame
    def push_frame(self, frame_bgr: np.ndarray):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        self._buffer.append(rgb)

    def maybe_encode(self):
        """Kick a background encode if one isn't already running and we have a
        full clip. Non-blocking; returns immediately."""
        if self._busy or len(self._buffer) < self.clip_frames:
            return
        clip = list(self._buffer)
        self._busy = True
        threading.Thread(target=self._work, args=(clip,), daemon=True).start()

    def _work(self, clip: list[np.ndarray]):
        try:
            if not self._loaded:
                with self._lock:
                    self.state.status = "loading"
                self.encoder.load()
                self._loaded = True
            emb = self.encoder.encode(clip)
            self._ingest(emb)
        except Exception as e:  # never let the worker kill the app
            print(f"[world_model] error: {e}")
        finally:
            self._busy = False

    def _ingest(self, emb: np.ndarray):
        with self._lock:
            if len(self._normal) < self.calib_clips:
                # Calibration: assume the operator shows normal/safe operation.
                self._normal.append(emb)
                self.state.status = "calibrating"
                self.state.progress = f"{len(self._normal)}/{self.calib_clips}"
                if len(self._normal) == self.calib_clips:
                    self._fit_baseline()
                return
            # Monitoring: score drift from the normal manifold.
            self._score(emb)

    def _fit_baseline(self):
        X = np.stack(self._normal)
        self._mean = X.mean(axis=0)
        dists = np.linalg.norm(X - self._mean, axis=1)
        self._dist_mu = float(dists.mean())
        self._dist_sigma = float(dists.std() + 1e-6)
        try:
            from sklearn.decomposition import PCA

            self._pca = PCA(n_components=2).fit(X)
            self.state.normal_2d = self._pca.transform(X)
        except Exception:
            self._pca = None
        self.state.status = "monitoring"

    def _score(self, emb: np.ndarray):
        d = float(np.linalg.norm(emb - self._mean))
        z = (d - self._dist_mu) / self._dist_sigma
        self.state.danger_score = z
        self.state.is_anomaly = z > self.z_threshold
        if self._pca is not None:
            p = self._pca.transform(emb[None, :])[0]
            self.state.recent_2d.append((float(p[0]), float(p[1])))
            self.state.recent_2d = self.state.recent_2d[-30:]

    def snapshot(self) -> LatentState:
        with self._lock:
            # shallow copy of the mutable bits for thread-safe overlay reads
            s = LatentState(
                status=self.state.status,
                progress=self.state.progress,
                danger_score=self.state.danger_score,
                is_anomaly=self.state.is_anomaly,
                normal_2d=self.state.normal_2d,
                recent_2d=list(self.state.recent_2d),
            )
        return s
