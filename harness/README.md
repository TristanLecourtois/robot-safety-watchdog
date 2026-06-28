# Safety Harness

Runtime safety layer for OpenARM/LeRobot pilots.

Core path:

```text
camera frame -> watchdog scene_context -> deterministic policy -> robot pause/hold/resume
```

The VLA/VLM is not required for interruption. Vision can be YOLO/YOLOE; the
harness decision is deterministic and auditable.

## Package Layout

- `core.py`: `safe_execute` command gating.
- `runtime.py`: `RuntimeWatchdogSupervisor` for pause/resume while motion is already running.
- `policy.py`: deterministic `ALLOW`, `BLOCK`, `PAUSE`, `STOP`, `RESUME` decisions.
- `robot.py`: OpenARM/LeRobot adapters and hold-position controller shim.
- `perception.py`: adapter from watchdog outputs to harness `scene_context`.
- `logger.py`: JSONL audit logging.
- `scripts/`: hardware and hackathon demo entry points.
- `RUNBOOK.md`: commands for the OpenARM hackathon demos.
- `OPENARM_INTEGRATION.md`: longer integration notes and API expectations.

## Demo Scripts

Preferred module paths:

```bash
python3 -m harness.scripts.openarm_smoke
python3 -m harness.scripts.openarm_pause_resume
python3 -m harness.scripts.openarm_replay_watchdog
```

Backward-compatible wrappers also exist at:

```bash
python3 -m harness.openarm_smoke
python3 -m harness.openarm_pause_resume
python3 -m harness.openarm_replay_watchdog
```

For the tested OpenARM commands, see [RUNBOOK.md](RUNBOOK.md).

Minimal OpenARM gating shape:

```python
from harness import OpenArmLeRobotAdapter, WatchdogPerceptionAdapter, safe_execute

perception = WatchdogPerceptionAdapter(camera_id="external_webcam_1")
robot = OpenArmLeRobotAdapter(controller=openarm_controller)

decision = safe_execute(
    {
        "id": "act_001",
        "type": "pick",
        "object": "sharp_tool",
        "arm": "left_arm",
        "source": "teleop_replay",
        "trajectory": replay_trajectory,
    },
    camera=cv2.VideoCapture(0),
    perception_adapter=perception,
    robot_adapter=robot,
)
```

The adapter is intentionally duck-typed because LeRobot/OpenARM setups differ.
Expose one of these methods on the controller: `execute`, `run_action`,
`send_action`, `play_trajectory`, or `replay_trajectory`. For runtime protection,
provide `pause`, `resume`, and preferably `stop` or `emergency_stop`.

`BLOCK` and `ALLOW` are command-gating decisions before motion is sent. For
interruption while OpenARM is already moving, use `PAUSE`, `STOP`, and `RESUME`.
