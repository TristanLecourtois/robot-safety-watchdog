"""The watchdog loop — wires the layers together.

Every frame:  detect (masks) -> hands -> rule engine (precise geometry).
On a rule hit (or on an idle timer): escalate to the VLM for a judgment.
Alerts are logged to JSONL (the "insurance data") and surfaced on the overlay.
"""
from __future__ import annotations

import json
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
        self.detector = Detector(config.yolo_model, config.thresholds.min_confidence)
        self.hands = HandTracker() if config.use_hand_landmarks else None
        self.rules = RuleEngine(config.thresholds)
        self.vlm = VLMJudge(config.vlm_model)
        self._last_vlm_t = 0.0
        self._last_alert_t = 0.0
        self._last_verdict: Verdict | None = None

    def process_frame(self, frame, now: float):
        detections = self.detector.detect(frame)
        hands = self.hands.detect(frame) if self.hands else []
        analysis = self.rules.analyze(detections, hands)

        sev = analysis.max_severity
        facts = self._facts_string(analysis)

        # Escalate to the VLM when a rule fires (rate-limited) or on the idle timer.
        want_vlm = (
            (sev and now - self._last_vlm_t >= self.cfg.vlm_min_interval_s)
            or (now - self._last_vlm_t >= self.cfg.vlm_idle_interval_s)
        )
        if want_vlm and self.vlm.available:
            self._last_vlm_t = now
            verdict = self.vlm.judge(frame, facts)
            if verdict is not None:
                self._last_verdict = verdict

        self._maybe_alert(analysis, self._last_verdict, facts, now)
        return detections, hands, analysis

    def _facts_string(self, analysis: FrameAnalysis) -> str:
        if not analysis.hits:
            return "No rule-level hazards detected this frame."
        return "\n".join(f"- [{h.severity}] {h.reason}" for h in analysis.hits)

    def _maybe_alert(self, analysis: FrameAnalysis, verdict, facts, now):
        rule_sev = analysis.max_severity
        vlm_critical = verdict is not None and verdict.severity in ("warning", "critical")
        if not (rule_sev or vlm_critical):
            return
        if now - self._last_alert_t < self.cfg.alert_cooldown_s:
            return
        self._last_alert_t = now
        event = {
            "ts": now,
            "rule_severity": rule_sev,
            "rule_hits": [asdict(h) for h in analysis.hits],
            "vlm_verdict": asdict(verdict) if verdict else None,
        }
        with open(self.cfg.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
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
            overlay.draw(frame, detections, hands, analysis, rationale, sev)
            cv2.imshow("Robot Safety Watchdog", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
