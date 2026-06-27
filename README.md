# Robot Safety Vision Watchdog

An **independent, external** vision watchdog for learning-based home robots. It
watches a robot operate (starting with the kitchen — 47% of household injuries)
and flags dangerous situations in real time, without trusting what the robot
*intends* to do. This is the "crash-test / SOC 2 for robotic safety" thesis made
concrete: we don't verify the model's weights, we **measure observed behavior**.

## Why this design

- **Empirical, not simulated.** Judgments come from the real camera feed.
- **Precise geometry, not just boxes.** A bounding box can't tell you which way a
  knife points. We segment the blade, recover its **orientation and tip** via PCA,
  and track **fingertips** with MediaPipe — so we can ask the question safety
  actually cares about: *is the blade tip close to and pointed at a hand?*
- **Generalist by design.** The VLM (Claude vision) is the open-vocabulary
  detector — it judges *any* dangerous situation (collision, hot liquid, fall,
  crush, child in workspace…), not a fixed list. It runs **continuously in the
  background (~1.5 s cadence)**. The deterministic rules are a **fast reflex
  layer** (<100 ms) for high-frequency hazards where sub-second reaction matters.
- **Not frame-rate real-time for the VLM — by design.** A vision API call takes
  1-4 s, so it can't run at 30 fps. The fast layer gives instant reflexes; the VLM
  runs async so it never stalls the loop, and the overlay shows its latest verdict.

## Architecture

```
 camera ─▶ Layer 1: Perception (every frame)
           ├─ YOLOE (open-vocab seg) → masks + classes for ANY prompted object
           │    (knife, cleaver, stove, flame, "hand", … — edit the prompt list)
           └─ MediaPipe → 21 hand landmarks (where wheels exist; else "hand" prompt)
                 │
                 ▼
           orientation.py → blade axis / tip / angle (PCA on mask)
                 │
                 ▼
 Layer 1.5: Rule engine (deterministic, real-time)
   • blade tip → nearest fingertip distance
   • blade aim angle (is it pointing AT the hand?)
   • fragile object fast-motion (drop risk)
   • object near hot zone (oven/stove)
                 │  on hit (or idle timer)
                 ▼
 Layer 2: VLM judge (Claude vision) — GENERALIST, runs continuously, async
   • open-vocabulary: any hazard, not a fixed list
   • returns: dangerous? severity? category? rationale? recommended_action?
                 │
                 ▼
        JSONL event log  +  live overlay  (+ kill-switch hook)
```

| File | Role |
|------|------|
| [config.py](config.py) | All thresholds and model choices |
| [src/detector.py](src/detector.py) | YOLO segmentation + tracking → `Detection` |
| [src/orientation.py](src/orientation.py) | **Blade orientation & tip via PCA on the mask** |
| [src/pose.py](src/pose.py) | MediaPipe hand landmarks / fingertips |
| [src/rules.py](src/rules.py) | Deterministic safety rules over the fine geometry |
| [src/vlm_judge.py](src/vlm_judge.py) | Claude vision danger verdict (structured output) |
| [src/overlay.py](src/overlay.py) | Debug/demo visualization |
| [src/watchdog.py](src/watchdog.py) | Main loop wiring the layers together |

## Setup & Run (uv — recommended)

[uv](https://docs.astral.sh/uv/) handles the virtualenv, the dependencies, and
even fetches a compatible Python (MediaPipe has no 3.13 wheels yet, so the
project pins `>=3.10,<3.13` and uv grabs the right interpreter automatically).

```bash
cp .env.example .env        # then add your ANTHROPIC_API_KEY
uv run main.py              # live webcam — installs deps on first run
uv run main.py --video clip.mp4   # analyze a recorded clip
```

`uv run` creates `.venv` and installs everything the first time, then is instant.
No manual `activate` needed. Press `q` to quit. Alerts stream to the console and
to `watchdog_events.jsonl`.

<details>
<summary>Alternative: plain pip</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
</details>

> MediaPipe is optional — if it isn't installed the system falls back to
> person-box proximity instead of fingertip precision.

## Tuning precision

Everything lives in [config.py](config.py):
- `detector_backend` — `"yoloe"` (open-vocabulary segmentation, default) or
  `"yolo"` (classic COCO seg). YOLOE segments/classifies whatever you list in
  `OPEN_VOCAB_PROMPTS`, so adding a new hazard is just adding a word.
- `OPEN_VOCAB_PROMPTS` — the things to detect (sharp/fragile/hot/person/hand/…).
  Add "cleaver", "boiling water", "robot gripper", etc. and the hazard category
  sets below it pick them up.
- `blade_tip_to_hand_px` — how close the tip must be to a hand to fire.
- `blade_aim_angle_deg` — how directly the blade must point at the hand (smaller =
  must be aimed straight at it) to escalate to *critical*.
- `yoloe_model` / `yolo_model` — `yoloe-11s-seg` → `11m`/`11l` (or
  `yolo11l-seg.pt` → `yolo11x-seg.pt`) for more accuracy at some fps cost.

> First YOLOE run pulls Ultralytics' CLIP and MobileCLIP (~572 MB, one time) to
> embed the text prompts; embeddings are then cached to `.textpe_cache.pt`.
- `vlm_model` — `claude-opus-4-8` (best), `claude-sonnet-4-6` (faster),
  `claude-haiku-4-5` (cheapest) for higher fps on the reasoning layer.

## Notes & next steps

- Distances are in **pixels** (image space). For a real test rig, calibrate to cm
  with a known-size reference object or a depth camera.
- COCO has no "stove"/"robot arm" classes; we proxy with `oven`/`microwave` and
  treat held sharp/fragile objects as the manipulated item. A fine-tuned model on
  kitchen + robot-arm classes is the obvious upgrade.
- The `recommended_action` field (`stop`/`slow_down`/...) is the natural hook for a
  real kill-switch / robot-control integration.
