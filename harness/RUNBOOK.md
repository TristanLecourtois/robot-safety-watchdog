# OpenARM Harness Runbook

This is the hackathon-facing runbook for the OpenARM safety harness demos.

The harness has two proof paths:

- **Control proof**: OpenARM moves, the harness calls `pause`, holds pose, resumes, then stops.
- **Safety proof**: a LeRobot recorded episode is replayed, the camera watchdog detects danger, and the harness pauses the replay mid-motion.

The VLA/VLM is not required for the safety proof. The interruption path is:

```text
camera frame -> YOLO/watchdog rules -> deterministic harness policy -> pause/hold/resume
```

## 1. Environment

Activate the LeRobot environment and expose both repos:

```bash
cd /home/rached/openarm/lerobot
source .venv/bin/activate

cd /home/rached/robot-safety-watchdog
export PYTHONPATH=/home/rached/openarm/lerobot/src:/home/rached/robot-safety-watchdog:$PYTHONPATH
```

Install demo-only visualization support if needed:

```bash
python3 -m pip install rerun-sdk
```

If the LeRobot environment does not already include the watchdog dependencies:

```bash
python3 -m pip install python-dotenv opencv-python ultralytics anthropic lap
```

## 2. Hardware Checks

Bring CAN up before connecting OpenARM:

```bash
python3 -m lerobot.scripts.lerobot_setup_can --mode=setup --interfaces=can0,can1
```

Find cameras:

```bash
v4l2-ctl --list-devices
```

Preview the external webcam:

```bash
ffplay /dev/video4
```

## 3. Control Proof: Pause / Resume / Stop

This does not use perception. It validates that the harness can hold and resume OpenARM.

```bash
OPENARM_MODE=bi \
OPENARM_ID=my_biopenarm \
OPENARM_LEFT_PORT=can0 \
OPENARM_RIGHT_PORT=can1 \
python3 -m harness.scripts.openarm_pause_resume \
  --robot-factory harness.scripts.openarm_robot_factory:build_robot \
  --connect \
  --motion \
  --motion-joint left_joint_1.pos \
  --motion-amplitude-deg 6 \
  --motion-period 6 \
  --telemetry \
  --before-pause 10 \
  --before-resume 10 \
  --before-stop 10 \
  --stop-mode hold
```

Expected behavior:

- OpenARM makes a small motion.
- `pause` holds the current pose.
- `resume` allows motion again.
- `stop` holds the final pose.

## 4. Safety Proof: Replay Paused By Danger

Use a local LeRobot dataset root containing `data/`, `meta/`, and `videos/`.

Example dataset:

```text
/home/rached/Téléchargements/cut_20260628_042259-20260628T025110Z-3-001/cut_20260628_042259
```

Fast diagnostic mode: pause when a human is detected.

```bash
OPENARM_MODE=bi \
OPENARM_ID=my_biopenarm \
OPENARM_LEFT_PORT=can0 \
OPENARM_RIGHT_PORT=can1 \
python3 -m harness.scripts.openarm_replay_watchdog \
  --robot-factory harness.scripts.openarm_robot_factory:build_robot \
  --dataset-repo-id local/cut_20260628_042259 \
  --dataset-root "/home/rached/Téléchargements/cut_20260628_042259-20260628T025110Z-3-001/cut_20260628_042259" \
  --episode 0 \
  --connect \
  --camera-index /dev/video4 \
  --frame-width 640 \
  --frame-height 480 \
  --pause-on human-presence \
  --unsafe-frames-before-pause 4 \
  --clear-frames-before-resume 5 \
  --detector-backend yolo \
  --yolo-model yolo11n-seg.pt \
  --min-confidence 0.45 \
  --camera-buffer-size 1 \
  --drop-stale-camera-frames 2 \
  --no-hand-landmarks \
  --print-watchdog-every 1 \
  --stop-at-end \
  --rerun
```

Final danger mode: pause only when a sharp object is associated with a human hand/person.

```bash
  --pause-on sharp-hand \
  --sharp-hand-proximity-px 120
```

Use `hand-presence` when demonstrating hand-triggered pause with open-vocabulary or hand landmarks.

## 5. Rerun Visualization

With `--rerun`, the script logs:

```text
camera/image
camera/detections
watchdog/mode
watchdog/decision
watchdog/hazards
```

Save a replayable Rerun file instead of opening the viewer:

```bash
--rerun --rerun-mode save --rerun-save openarm_watchdog_replay.rrd
```

## 6. Latency Tuning

Use these settings for lower latency:

```bash
--detector-backend yolo
--yolo-model yolo11n-seg.pt
--frame-width 640
--frame-height 480
--camera-buffer-size 1
--drop-stale-camera-frames 2
--clear-frames-before-resume 5
```

If detections are noisy:

```bash
--unsafe-frames-before-pause 4
--min-confidence 0.45
```

If detections are missed:

```bash
--min-confidence 0.25
--yolo-model yolo11s-seg.pt
```

## 7. Generated Files

Do not commit generated logs or downloaded models:

- `*.jsonl`
- `*.rrd`
- `*.pt`
- `.textpe_cache.pt`
- `__pycache__/`

These are ignored by `.gitignore`.
