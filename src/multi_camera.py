"""Multi-camera watchdog — N parallel streams with cross-camera fusion.

Architecture
------------
Each camera runs in its own background thread:
    capture → Detector → RuleEngine → (per-camera VLM async)

The main loop reads the latest results from every stream, then:
  1. Correlates objects across cameras (same label seen in multiple views).
  2. Triangulates 3-D positions when camera calibration is provided.
  3. Dispatches a holistic multi-camera VLM call asynchronously.
  4. Renders a tiled grid display with per-camera overlays + summary strip.

Single-camera fallback
----------------------
`run_multicam` with one camera degrades gracefully to standard single-camera
behaviour without the multi-camera VLM or triangulation overhead.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

import config as cfg
from src import overlay
from src.detector import Detection
from src.pose import Hand
from src.rules import FrameAnalysis
from src.vlm_judge import VLMJudge, Verdict
from src.watchdog import Watchdog


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrossCameraObject:
    """An object class observed in two or more camera streams."""
    label: str
    cameras: list[str]                                 # which camera labels see it
    centers_px: dict[str, tuple[float, float]] = field(default_factory=dict)  # cam→image center
    position_3d: tuple[float, float, float] | None = None  # triangulated world coords (m)
    confidence: float = 1.0


@dataclass
class MultiCameraAnalysis:
    """Fused result for one display tick."""
    per_camera: dict[str, tuple[list[Detection], list[Hand], FrameAnalysis]] = \
        field(default_factory=dict)
    cross_camera_objects: list[CrossCameraObject] = field(default_factory=list)
    multi_verdict: Verdict | None = None

    @property
    def max_severity(self) -> str | None:
        sev = set()
        for _, _, fa in self.per_camera.values():
            if fa.max_severity:
                sev.add(fa.max_severity)
        if self.multi_verdict and self.multi_verdict.severity in ("warning", "critical"):
            sev.add(self.multi_verdict.severity)
        if "critical" in sev:
            return "critical"
        if "warning" in sev:
            return "warning"
        return None


# ---------------------------------------------------------------------------
# Triangulation (DLT — requires calibrated cameras)
# ---------------------------------------------------------------------------

def _parse_calibration(cam_cfg: cfg.CameraConfig):
    """Return (K, R, t) numpy arrays or None if calibration is missing."""
    if cam_cfg.intrinsics is None or cam_cfg.extrinsics is None:
        return None
    K = np.array(cam_cfg.intrinsics, dtype=np.float64).reshape(3, 3)
    E = np.array(cam_cfg.extrinsics, dtype=np.float64).reshape(4, 4)
    R, t = E[:3, :3], E[:3, 3]
    return K, R, t


def _triangulate_dlt(
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray, pt1: tuple[float, float],
    K2: np.ndarray, R2: np.ndarray, t2: np.ndarray, pt2: tuple[float, float],
) -> np.ndarray | None:
    """Linear triangulation (DLT) for one 2-D correspondence → 3-D world point."""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    u1, v1 = pt1
    u2, v2 = pt2
    A = np.array([
        v1 * P1[2] - P1[1],
        P1[0] - u1 * P1[2],
        v2 * P2[2] - P2[1],
        P2[0] - u2 * P2[2],
    ], dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return (X[:3] / X[3]).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-camera stream (background thread)
# ---------------------------------------------------------------------------

class CameraStream:
    """Captures and processes one camera in a daemon thread.

    The thread writes (frame, detections, hands, analysis) atomically under a
    lock. The main loop reads the latest snapshot without blocking the capture.
    """

    def __init__(self, cam_cfg: cfg.CameraConfig, shared_cfg: cfg.WatchdogConfig):
        self.cam_cfg = cam_cfg
        # Build a per-camera WatchdogConfig inheriting all shared settings.
        pcfg = cfg.WatchdogConfig(
            camera_index=cam_cfg.source if isinstance(cam_cfg.source, int) else 0,
            frame_width=cam_cfg.frame_width,
            frame_height=cam_cfg.frame_height,
            detector_backend=shared_cfg.detector_backend,
            yoloe_model=shared_cfg.yoloe_model,
            open_vocab_prompts=list(shared_cfg.open_vocab_prompts),
            yolo_model=shared_cfg.yolo_model,
            textpe_cache=shared_cfg.textpe_cache,
            use_hand_landmarks=shared_cfg.use_hand_landmarks,
            vlm_model=shared_cfg.vlm_model,
            vlm_interval_s=shared_cfg.vlm_interval_s,
            vlm_min_interval_s=shared_cfg.vlm_min_interval_s,
            vlm_async=shared_cfg.vlm_async,
            alert_cooldown_s=shared_cfg.alert_cooldown_s,
            log_path=shared_cfg.log_path,
            thresholds=shared_cfg.thresholds,
        )
        self.watchdog = Watchdog(pcfg)
        self._latest: tuple[np.ndarray, list[Detection], list[Hand], FrameAnalysis] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{self.cam_cfg.label}", daemon=True
        )
        self._thread.start()
        print(f"[{self.cam_cfg.label}] stream started (source={self.cam_cfg.source!r})")

    def stop(self):
        self._stop.set()

    def get_latest(self) -> tuple[np.ndarray, list[Detection], list[Hand], FrameAnalysis] | None:
        with self._lock:
            return self._latest

    def verdict_banner(self) -> tuple[str | None, str | None]:
        return self.watchdog.verdict_banner()

    def _loop(self):
        source = self.cam_cfg.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[{self.cam_cfg.label}] ERROR: could not open source {source!r}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.frame_height)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                print(f"[{self.cam_cfg.label}] stream ended or read failed")
                break
            now = time.time()
            detections, hands, analysis = self.watchdog.process_frame(frame, now)
            with self._lock:
                self._latest = (frame.copy(), detections, hands, analysis)
        cap.release()


# ---------------------------------------------------------------------------
# Multi-camera orchestrator
# ---------------------------------------------------------------------------

class MultiCameraWatchdog:
    """Runs N CameraStreams in parallel and fuses their results each tick."""

    def __init__(self, cam_cfgs: list[cfg.CameraConfig], shared_cfg: cfg.WatchdogConfig):
        self.shared_cfg = shared_cfg
        self.cam_cfgs = cam_cfgs
        self.streams = [CameraStream(cc, shared_cfg) for cc in cam_cfgs]
        # Separate VLMJudge for the holistic multi-camera call.
        self.multi_vlm = VLMJudge(shared_cfg.vlm_model)
        self._last_multi_vlm_t = 0.0
        self._last_multi_verdict: Verdict | None = None
        self._multi_vlm_busy = False
        self._lock = threading.Lock()

    def start(self):
        for stream in self.streams:
            stream.start()

    def stop(self):
        for stream in self.streams:
            stream.stop()

    # ----- main-loop call ----------------------------------------------------

    def fuse(self, now: float) -> tuple[list[np.ndarray | None], MultiCameraAnalysis]:
        """Collect latest per-camera results, run cross-camera fusion, return display data."""
        frames: list[np.ndarray | None] = []
        analysis = MultiCameraAnalysis()

        for stream, cc in zip(self.streams, self.cam_cfgs):
            latest = stream.get_latest()
            if latest is None:
                frames.append(None)
                analysis.per_camera[cc.label] = ([], [], FrameAnalysis())
            else:
                frame, dets, hands, fa = latest
                frames.append(frame)
                analysis.per_camera[cc.label] = (dets, hands, fa)

        analysis.cross_camera_objects = self._correlate(analysis.per_camera)
        self._try_triangulate(analysis)

        # Multi-camera VLM: only worth dispatching when ≥2 cameras have frames.
        valid_frames = [(f, cc.label) for f, cc in zip(frames, self.cam_cfgs) if f is not None]
        if (
            len(valid_frames) >= 2
            and self.multi_vlm.available
            and not self._multi_vlm_busy
            and now - self._last_multi_vlm_t >= self.shared_cfg.vlm_interval_s * 2.0
        ):
            self._last_multi_vlm_t = now
            self._dispatch_multi_vlm(
                [f for f, _ in valid_frames],
                [lbl for _, lbl in valid_frames],
                analysis,
            )

        with self._lock:
            analysis.multi_verdict = self._last_multi_verdict

        return frames, analysis

    # ----- cross-camera correlation ------------------------------------------

    def _correlate(
        self,
        per_camera: dict[str, tuple[list[Detection], list[Hand], FrameAnalysis]],
    ) -> list[CrossCameraObject]:
        """Group object classes seen in more than one camera stream."""
        label_data: dict[str, CrossCameraObject] = {}
        for cam_label, (dets, _, _) in per_camera.items():
            for d in dets:
                if d.label not in label_data:
                    label_data[d.label] = CrossCameraObject(
                        label=d.label, cameras=[], confidence=d.confidence
                    )
                cco = label_data[d.label]
                if cam_label not in cco.cameras:
                    cco.cameras.append(cam_label)
                cco.centers_px[cam_label] = d.center
                cco.confidence = max(cco.confidence, d.confidence)
        return [cco for cco in label_data.values() if len(cco.cameras) > 1]

    # ----- triangulation -----------------------------------------------------

    def _try_triangulate(self, analysis: MultiCameraAnalysis):
        """Fill in CrossCameraObject.position_3d for calibrated camera pairs."""
        calibrated = [
            (cc, _parse_calibration(cc))
            for cc in self.cam_cfgs
            if cc.intrinsics is not None and cc.extrinsics is not None
        ]
        calibrated = [(cc, cal) for cc, cal in calibrated if cal is not None]
        if len(calibrated) < 2:
            return

        for cco in analysis.cross_camera_objects:
            if cco.position_3d is not None:
                continue
            # Find two calibrated cameras that both see this object.
            views = [
                (cc, cal) for cc, cal in calibrated
                if cc.label in cco.cameras and cc.label in cco.centers_px
            ]
            if len(views) < 2:
                continue
            (cc1, (K1, R1, t1)), (cc2, (K2, R2, t2)) = views[0], views[1]
            pt3d = _triangulate_dlt(
                K1, R1, t1, cco.centers_px[cc1.label],
                K2, R2, t2, cco.centers_px[cc2.label],
            )
            if pt3d is not None:
                cco.position_3d = (float(pt3d[0]), float(pt3d[1]), float(pt3d[2]))

    # ----- multi-camera VLM --------------------------------------------------

    def _dispatch_multi_vlm(
        self,
        frames: list[np.ndarray],
        cam_labels: list[str],
        analysis: MultiCameraAnalysis,
    ):
        facts_lines: list[str] = []
        for cam_label, (_, _, fa) in analysis.per_camera.items():
            for h in fa.hits:
                facts_lines.append(f"[{cam_label}] [{h.severity}] {h.reason}")
        for cco in analysis.cross_camera_objects:
            cams_str = ", ".join(cco.cameras)
            if cco.position_3d:
                x, y, z = cco.position_3d
                facts_lines.append(
                    f"'{cco.label}' seen in [{cams_str}]; "
                    f"triangulated 3-D position ({x:.2f}, {y:.2f}, {z:.2f}) m"
                )
            else:
                facts_lines.append(f"'{cco.label}' seen in both [{cams_str}]")
        facts = "\n".join(facts_lines) or "No rule-level hazards detected across any camera."

        def work():
            verdict = self.multi_vlm.judge_multi(frames, cam_labels, facts)
            with self._lock:
                if verdict is not None:
                    self._last_multi_verdict = verdict
                self._multi_vlm_busy = False
            if verdict is not None and verdict.severity in ("warning", "critical"):
                _log_multi_event(verdict, facts, time.time(), self.shared_cfg.log_path)

        self._multi_vlm_busy = True
        if self.shared_cfg.vlm_async:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_multi_event(verdict: Verdict, facts: str, now: float, log_path: str):
    event = {
        "ts": now,
        "source": "multi_camera",
        "vlm_verdict": asdict(verdict),
        "cross_camera_facts": facts,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[MULTI-CAM ALERT] vlm={verdict.severity} action={verdict.recommended_action}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_multicam(config: cfg.WatchdogConfig):
    """Main loop for multi-camera mode."""
    cam_cfgs = config.camera_configs

    if len(cam_cfgs) == 1:
        # Degrade to single-camera path — avoid overhead of multi-camera VLM.
        from src.watchdog import run_webcam
        config.camera_index = cam_cfgs[0].source if isinstance(cam_cfgs[0].source, int) else 0
        run_webcam(config)
        return

    mwd = MultiCameraWatchdog(cam_cfgs, config)
    mwd.start()

    # Wait briefly for streams to produce first frames.
    time.sleep(1.0)

    # Estimate cell size from first available frame, cap to reasonable display.
    _cell_w, _cell_h = 640, 360

    print(f"Multi-camera watchdog running ({len(cam_cfgs)} cameras). Press 'q' to quit.")
    try:
        while True:
            now = time.time()
            frames, analysis = mwd.fuse(now)

            # Build per-camera display lists.
            cam_labels, dets_list, hands_list, fa_list, rat_list, sev_list = [], [], [], [], [], []
            for cc in cam_cfgs:
                cam_labels.append(cc.label)
                dets, hands, fa = analysis.per_camera.get(cc.label, ([], [], FrameAnalysis()))
                dets_list.append(dets)
                hands_list.append(hands)
                fa_list.append(fa)
                stream = next(s for s in mwd.streams if s.cam_cfg.label == cc.label)
                rat, vlm_sev = stream.verdict_banner()
                per_sev = fa.max_severity or vlm_sev
                rat_list.append(rat)
                sev_list.append(per_sev)

            mv = analysis.multi_verdict
            multi_rat = mv.rationale if mv else None
            multi_sev = analysis.max_severity

            grid = overlay.draw_grid(
                frames=frames,
                cam_labels=cam_labels,
                detections_list=dets_list,
                hands_list=hands_list,
                analyses_list=fa_list,
                rationales_list=rat_list,
                severities_list=sev_list,
                cross_objects=analysis.cross_camera_objects,
                multi_verdict_text=multi_rat,
                multi_severity=multi_sev,
                cell_w=_cell_w,
                cell_h=_cell_h,
            )

            cv2.imshow("Robot Safety Watchdog — Multi-Camera", grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        mwd.stop()
        cv2.destroyAllWindows()
