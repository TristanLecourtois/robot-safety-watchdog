"""Generate scene-describing event logs with Claude vision (VLM).

Samples frames from each camera video, asks Claude to write a terse safety
event-log line per frame, and writes logs.json (a timeline of
{t, cam, severity, msg}). The dashboard replays these synced to the video, so
the log feed genuinely describes what's on screen.

    uv run python scripts/generate_vlm_logs.py            # defaults
    uv run python scripts/generate_vlm_logs.py --step 8 --model claude-sonnet-4-6

One-time, before the demo. Needs ANTHROPIC_API_KEY (already in .env).
"""
from __future__ import annotations

import argparse
import base64
import json

import cv2
from dotenv import load_dotenv

load_dotenv()
import anthropic  # noqa: E402

# (video file, camera label as shown on the dashboard)
FEEDS = [
    ("WIN_20260627_21_00_58_Pro.mp4", "SCENE CAM"),
    ("gripper.mp4", "GRIPPER CAM"),
    ("head.mp4", "HEAD CAM"),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        "message": {"type": "string"},
    },
    "required": ["severity", "message"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a robot-safety watchdog writing a live event log from a single "
    "camera. Given one frame, output ONE terse log line (max ~70 chars) that "
    "describes what is actually visible and flags any hazard. Be concrete about "
    "the objects, hands, and actions you see. Severity: 'critical' only for "
    "imminent danger (a sharp blade near a hand, an imminent collision, hot "
    "contact, something about to fall/spill); 'warning' for risky-but-not-yet; "
    "'info' for normal observation. No preamble, just the line."
)


def frame_at(cap, t_s):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, f = cap.read()
    return f if ok else None


def describe(client, model, label, frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    b64 = base64.standard_b64encode(buf.tobytes()).decode()
    try:
        resp = client.messages.create(
            model=model, max_tokens=300, system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": f"Camera: {label}. Write the event-log line for this frame."},
            ]}],
        )
    except Exception as e:
        print(f"  ! {label}: {e}")
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    try:
        d = json.loads(text)
        return {"severity": d["severity"], "msg": d["message"][:90]}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=8.0, help="seconds between samples")
    ap.add_argument("--max-seconds", type=float, default=260.0, help="cover up to this (master clip length)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--out", default="logs.json")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    entries = [{"t": 0.0, "cam": "SYSTEM", "severity": "info",
                "msg": "Watchdog online — monitoring 3 feeds"}]

    for path, label in FEEDS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"skip {path} (cannot open)")
            continue
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        end = min(args.max_seconds, dur)
        t = 1.0
        print(f"{label}: sampling 0-{end:.0f}s every {args.step}s ...")
        while t < end:
            frame = frame_at(cap, t)
            if frame is not None:
                d = describe(client, args.model, label, frame)
                if d:
                    entries.append({"t": round(t, 1), "cam": label, **d})
                    print(f"  [{t:5.1f}s] {label:11s} {d['severity']:8s} {d['msg']}")
            t += args.step
        cap.release()

    entries.sort(key=lambda e: e["t"])
    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nwrote {args.out} — {len(entries)} log entries")


if __name__ == "__main__":
    main()
