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
    "You are a GENERALIST safety watchdog observing a learning-based robot "
    "operating in a home. You are an INDEPENDENT external observer — do not trust "
    "what the robot intends; judge only the observed scene. Assess danger to "
    "PEOPLE first, then pets, then property.\n\n"
    "Consider the full open-ended space of hazards, not a fixed list — for "
    "example: sharp objects near a person; the robot or an object on a collision "
    "course with a person, child, or pet; hot surfaces, flames, or boiling/hot "
    "liquid about to be spilled or touched; heavy or fragile objects about to "
    "fall, tip, or shatter; an unstable or over-tight grip; the robot near a "
    "stair/counter edge or about to crush a limb; electrical/water hazards; a "
    "child reaching into the robot's workspace. If something looks unsafe but "
    "doesn't fit any category, still flag it and describe it.\n\n"
    "You may be given precise geometric facts already measured by a vision "
    "pipeline (e.g. a blade's orientation and tip-to-fingertip distance). Treat "
    "them as helpful hints, NOT as the limit of what to look for — judge the whole "
    "scene. Be decisive and calibrated: reserve 'critical' for imminent harm. "
    "Reply only via the structured format."
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
