# Safety Harness Package

This package is the runtime harness layer around the existing watchdog code.

Current implementation:

- `WatchdogPerceptionAdapter` calls `src.watchdog.Watchdog.process_frame(frame)`.
- `PolicyEngine` converts the resulting scene context into `ALLOW`, `BLOCK`, `PAUSE`, `STOP`, or `RESUME`.
- `OpenArmLeRobotAdapter` gates real OpenARM control through an injected LeRobot/OpenARM controller.
- `LeRobotOpenArmController` adapts current LeRobot `main` OpenARM robots by using `send_action`, `get_observation`, inference `pause/resume`, and hold-position actions.
- `safe_execute` is the command-gating entry point: unsafe decisions skip the robot action.
- `RuntimeWatchdogSupervisor` is the runtime loop helper: keep cameras running, pause/stop on danger, resume after safe frames.
- `JsonlAuditLogger` writes decision evidence to `harness_events.jsonl`.

For OpenARM-specific setup details, see [OPENARM_INTEGRATION.md](OPENARM_INTEGRATION.md).

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
