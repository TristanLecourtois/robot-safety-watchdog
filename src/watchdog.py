"""The watchdog loop — wires the layers together.

Every frame:  detect (masks) -> hands -> rule engine (precise geometry).
On a rule hit (or on an idle timer): escalate to the VLM for a judgment.
Alerts are logged to JSONL (the "insurance data") and surfaced on the overlay.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict

import cv2

import config as cfg
from src import overlay
from src.detector import Detector
from src.pose import HandTracker
from src.rules import FrameAnalysis, RuleEngine
from src.vlm_judge import VLMJudge, Verdict


class Watchdog:
    def __init__(self, config: cfg.WatchdogConfig = cfg.CONFIG):
        self.cfg = config
        self.detector = Detector.build(config)
        self.hands = HandTracker() if config.use_hand_landmarks else None
        self.rules = RuleEngine(config.thresholds)
        self.vlm = VLMJudge(config.vlm_model)
        self._last_vlm_t = 0.0
        self._last_alert_t = 0.0
        self._last_verdict: Verdict | None = None
        self._vlm_busy = False           # a background judgment is in flight
        self._lock = threading.Lock()

        # Optional world-model track (lazy, heavy). Built only when enabled.
        self.world = None
        if config.enable_world_model:
            from src.world_model import LatentOODMonitor, VJEPAEncoder
            enc = VJEPAEncoder(config.wm_model, config.wm_input_size)
            self.world = LatentOODMonitor(
                enc, config.wm_clip_frames, config.wm_calib_clips,
                config.wm_z_threshold, config.wm_input_size,
            )

        # Optional generative future-preview track (GPU only).
        self.future = None
        if config.enable_future_preview:
            from src.future_preview import (AnchorDangerScorer, FutureFramePredictor,
                                            FuturePreviewMonitor)
            predictor = FutureFramePredictor(config.svd_model, config.future_num_frames)
            scorer = AnchorDangerScorer(config.anchors_dir)
            self.future = FuturePreviewMonitor(
                predictor, scorer, config.future_interval_s, config.future_danger_threshold,
            )

    def process_frame(self, frame, now: float):
        detections = self.detector.detect(frame)
        hands = self.hands.detect(frame) if self.hands else []
        analysis = self.rules.analyze(detections, hands)

        facts = self._facts_string(analysis)
        # GENERALIST cadence: run the VLM continuously. A rule hit just lets us
        # fire sooner (down to the min floor) so reaction is faster on known
        # hazards; with no hit we still judge the whole scene every interval.
        interval = self.cfg.vlm_min_interval_s if analysis.hits else self.cfg.vlm_interval_s
        if self.vlm.available and not self._vlm_busy and now - self._last_vlm_t >= interval:
            self._last_vlm_t = now
            self._dispatch_vlm(frame.copy(), facts)

        # World-model track: feed frames, kick a background encode when ready.
        if self.world is not None:
            self.world.push_frame(frame)
            self.world.maybe_encode()

        # Generative future-preview track (GPU): imagine the near future async.
        if self.future is not None:
            self.future.push_frame(frame)
            self.future.maybe_predict(now)

        self._maybe_alert(analysis, self._last_verdict, facts, now)
        return detections, hands, analysis

    def world_state(self):
        return self.world.snapshot() if self.world is not None else None

    def future_state(self):
        return self.future.snapshot() if self.future is not None else None

    def _dispatch_vlm(self, frame, facts: str):
        """Run the (slow) VLM call without blocking the fast loop."""
        def work():
            verdict = self.vlm.judge(frame, facts)
            with self._lock:
                if verdict is not None:
                    self._last_verdict = verdict
                self._vlm_busy = False
            if verdict is not None and verdict.severity in ("warning", "critical"):
                self._log_event(None, verdict, facts, time.time())

        self._vlm_busy = True
        if self.cfg.vlm_async:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def _facts_string(self, analysis: FrameAnalysis) -> str:
        if not analysis.hits:
            return "No rule-level hazards detected this frame."
        return "\n".join(f"- [{h.severity}] {h.reason}" for h in analysis.hits)

    def _maybe_alert(self, analysis: FrameAnalysis, verdict, facts, now):
        # Only the instant rule layer alerts from the fast loop; the VLM logs its
        # own verdicts from the background worker (see _dispatch_vlm).
        if not analysis.max_severity:
            return
        if now - self._last_alert_t < self.cfg.alert_cooldown_s:
            return
        self._last_alert_t = now
        self._log_event(analysis, verdict, facts, now)

    def _log_event(self, analysis: FrameAnalysis | None, verdict, facts, now):
        event = {
            "ts": now,
            "rule_severity": analysis.max_severity if analysis else None,
            "rule_hits": [asdict(h) for h in analysis.hits] if analysis else [],
            "vlm_verdict": asdict(verdict) if verdict else None,
        }
        with open(self.cfg.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        rule_sev = analysis.max_severity if analysis else None
        action = verdict.recommended_action if verdict else "monitor"
        print(f"[ALERT] rule={rule_sev} vlm={verdict.severity if verdict else '-'} action={action}")

    def verdict_banner(self):
        v = self._last_verdict
        if v is None:
            return None, None
        return v.rationale, (v.severity if v.severity in ("warning", "critical") else None)


def run_webcam(config: cfg.WatchdogConfig = cfg.CONFIG):
    wd = Watchdog(config)
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.camera_index}")

    print("Watchdog running. Press 'q' to quit.")
    try:
        while True:
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
