# VIGIE: Robot Safety Watchdog

VIGIE is an **external runtime safety layer** for learning-based robots. It
observes the robot and workspace through cameras, detects dangerous situations,
logs evidence, and can interrupt a real OpenARM/LeRobot run by pausing,
holding, and resuming the robot when the scene becomes safe again.

Built for the hackathon demo, the project combines three proof points:

- **See danger**: live vision detects hands, sharp tools, and unsafe spatial relations.
- **Decide deterministically**: auditable policies turn scene evidence into
  `ALLOW`, `BLOCK`, `PAUSE`, `STOP`, or `RESUME`.
- **Act on hardware**: the harness can pause a recorded OpenARM replay in the
  middle of execution, hold pose, and resume after fresh safe frames.

The thesis is simple: we do not claim to certify neural policy weights. We
measure observed behavior at runtime and provide the missing safety layer around
robot pilots.

## Hackathon demo in one line

**VIGIE watches a robot from the outside, detects unsafe scenes, and can pause a
real OpenARM/LeRobot replay before the robot keeps moving.**

What is demo-ready:

- **Live vision watchdog**: YOLO/YOLOE + deterministic geometry rules + optional
  Claude vision judgment.
- **OpenARM safety harness**: replay a recorded LeRobot episode, pause on camera
  danger, hold pose, then resume after the workspace clears.
- **Rerun visualization**: camera frame, detections, hazards, harness mode, and
  pause/resume decisions.
- **Web dashboard**: synchronized feeds, safety timeline, VLM logs, and
  future-preview artifacts from `main`.
- **Research tracks**: V-JEPA latent OOD monitor and Stable Video Diffusion
  future preview for “predict before harm” demos.

This is not a model-weight verifier or a certification claim. It is a runtime
evidence layer: observe the robot, make auditable deterministic safety decisions,
and intervene only through control surfaces the robot actually exposes.

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

 OpenARM harness path:
   LeRobot replay/action stream + watchdog scene_context
                 │
                 ▼
   deterministic harness policy: ALLOW / BLOCK / PAUSE / STOP / RESUME
                 │
                 ▼
   LeRobotOpenArmController: get_observation() -> send_action(current .pos)
   hold pose while paused, resume after fresh safe frames

 Parallel WORLD-MODEL track (optional, --worldmodel):
   V-JEPA 2 encodes short clips → latent space → OOD danger score
   • learns what "normal/safe" operation looks like (no danger labels)
   • flags moments that drift out of that distribution as surprising/unsafe
   • live 2D PCA map: green = normal cloud, moving dot = current situation
```

### World-model track (V-JEPA 2 latent OOD)

A self-supervised counterpart to the symbolic track: instead of naming hazards,
a video world model (V-JEPA 2) learns the *latent distribution of safe
operation* and flags out-of-distribution moments — "the model knows what normal
looks like; danger is surprise." Ties directly to the OOD/failure-mode thesis.

```bash
uv run --extra worldmodel main.py --worldmodel   # webcam + latent OOD panel
```

Workflow: show ~12 clips of normal/safe operation (auto-calibration), then it
monitors drift and lights up the latent map + danger z-score on anomalies.

> **CPU reality:** V-JEPA 2 ViT-L is heavy — ~80s one-time load and ~10-20s per
> clip on an Intel-Mac CPU. So this runs fully **async** as a periodic "deep
> glance" (a latent verdict every ~10-20s), never blocking the fast tracks. For
> a smooth demo, a GPU or precomputed clip latents are far better. One-time
> ~1.2GB model download. (transformers is pinned `<5` on Intel Mac: 5.x needs
> torch≥2.4, which has no macOS-Intel wheels.)

### Generative future-preview track (GPU — the "wow")

The generative half of the hybrid: from the current frame, **Stable Video
Diffusion imagines a short future clip**, we score each predicted frame for
danger via few-shot CLIP anchors, and raise a **preventive VETO** before
anything happens — "imagine the near future, veto danger." Camera-only,
automatic continuation (no robot action signal needed).

**GPU/CUDA only.** Set up the anchors (see [anchors/](anchors/)), then:

```bash
# Live: imagined-future strip + VETO under the main view
uv run --extra generative main.py --future

# Offline artifact (most reliable demo): a "present -> imagined future" MP4
uv run --extra generative python scripts/imagine_future.py \
    --image scene.jpg --anchors anchors/ --out imagined_future.mp4
```

> Stable Video Diffusion needs CUDA + fp16 and is heavy (seconds per clip even
> on GPU), so it runs **async** as a periodic glance. On the GPU box, install a
> **CUDA torch build** (the pins in `pyproject.toml` cap torch only on macOS
> Intel; elsewhere they float). The full pipeline (symbolic + VLM + V-JEPA
> latent + generative future) is the complete demo — run with
> `uv run --extra worldmodel --extra generative main.py --worldmodel --future`.

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
| [harness/](harness/) | Runtime robot safety harness for OpenARM/LeRobot |
| [harness/RUNBOOK.md](harness/RUNBOOK.md) | Hackathon commands for OpenARM pause/replay demos |
| [dashboard.py](dashboard.py) / [index.html](index.html) | Web dashboard and synchronized demo timeline |
| [scripts/](scripts/) | Future-preview, VLM-log, and Reachy helper scripts |

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

## OpenARM / LeRobot harness demo

The harness is the robot-control layer. It does not need a VLA policy. It can
wrap a LeRobot replay and pause it when the external camera watchdog sees a
dangerous scene.

Control-surface smoke test:

```bash
python3 -m harness.scripts.openarm_pause_resume \
  --robot-factory harness.scripts.openarm_robot_factory:build_robot \
  --connect --motion --stop-mode hold
```

Replay-with-danger demo:

```bash
python3 -m harness.scripts.openarm_replay_watchdog \
  --robot-factory harness.scripts.openarm_robot_factory:build_robot \
  --dataset-repo-id local/cut_20260628_042259 \
  --dataset-root "/path/to/lerobot_dataset_root" \
  --episode 0 \
  --connect \
  --camera-index /dev/video4 \
  --pause-on sharp-hand \
  --unsafe-frames-before-pause 4 \
  --clear-frames-before-resume 5 \
  --detector-backend yolo \
  --yolo-model yolo11n-seg.pt \
  --rerun
```

For the exact OpenARM commands used during the hackathon prep, see
[harness/RUNBOOK.md](harness/RUNBOOK.md).

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
