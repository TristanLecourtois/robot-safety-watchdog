"""Layer 2 — behavioral danger judgment with Claude vision.

The rule engine is precise but literal. The VLM adds judgment: it sees the
annotated frame plus the geometric facts we already computed (blade angle,
tip-to-finger distance, what's near the hot zone) and returns a structured
verdict with a severity, category, and a plain-language rationale. That
rationale is exactly the artifact an insurer / audit log wants.

Uses structured outputs (output_config.format) so we always get valid JSON.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import anthropic
    _SDK = True
except Exception:  # pragma: no cover
    _SDK = False


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "dangerous": {"type": "boolean"},
        "severity": {"type": "string", "enum": ["none", "low", "warning", "critical"]},
        "category": {"type": "string"},
        "rationale": {"type": "string"},
        "recommended_action": {
            "type": "string",
            "enum": ["none", "monitor", "slow_down", "stop"],
        },
    },
    "required": ["dangerous", "severity", "category", "rationale", "recommended_action"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a safety watchdog observing a learning-based robot operating in a "
    "home kitchen. You are an INDEPENDENT external observer — do not trust what "
    "the robot intends, judge only the observed scene. You are given an annotated "
    "frame and precise geometric facts already measured by a vision pipeline "
    "(blade orientation, blade-tip-to-fingertip distance in pixels, proximity to "
    "hot zones). Decide whether the current scene is dangerous to a human or "
    "would damage property. Be decisive: a sharp blade tip pointed at and close "
    "to a hand is critical. Reply only via the structured format."
)


@dataclass
class Verdict:
    dangerous: bool
    severity: str
    category: str
    rationale: str
    recommended_action: str

    @classmethod
    def from_json(cls, data: dict) -> "Verdict":
        return cls(
            dangerous=bool(data.get("dangerous", False)),
            severity=str(data.get("severity", "none")),
            category=str(data.get("category", "unknown")),
            rationale=str(data.get("rationale", "")),
            recommended_action=str(data.get("recommended_action", "none")),
        )


class VLMJudge:
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic() if _SDK else None

    @property
    def available(self) -> bool:
        return self.client is not None

    def judge(self, frame_bgr: np.ndarray, facts: str) -> Verdict | None:
        if self.client is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text":
                            "Measured geometric facts for this frame:\n" + facts +
                            "\n\nJudge the scene now."},
                    ],
                }],
            )
        except Exception as e:  # network/API hiccup shouldn't crash the watchdog
            print(f"[vlm] error: {e}")
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            return None
        try:
            return Verdict.from_json(json.loads(text))
        except json.JSONDecodeError:
            return None
