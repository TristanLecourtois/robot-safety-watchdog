# OpenARM / LeRobot Integration Notes

This is the minimal integration guide for plugging the harness into an OpenARM setup controlled through LeRobot.

LeRobot `main` inspection reference used for this guide:
`3dd19d043e2f3fe5673b13ea0ebe4f31884c0797`.

## 1. What Works Now

The current harness is easy to plug in for command gating:

- Wrap an OpenARM action, scripted move, teleop replay, or LeRobot action behind `safe_execute`.
- The harness captures a camera frame.
- The watchdog produces `scene_context`.
- `PolicyEngine` returns a decision.
- `BLOCK` skips the action.
- `ALLOW` sends the action to your OpenARM controller.

Runtime interruption is also supported by the adapter, but it depends on your controller exposing a real control method:

- `pause`
- `stop`
- `emergency_stop`
- `disconnect`
- a safe-hold command
- a neutral action that reliably stops motion

On LeRobot `main`, OpenARM robot classes expose `send_action`, `get_observation`,
and `disconnect`, but not a standard robot-level `pause` or `resume`. The harness
therefore provides `LeRobotOpenArmController`, a shim that maps pause/resume
semantics onto the current LeRobot APIs.

## 2. Expected Controller Surface

`OpenArmLeRobotAdapter` uses duck typing. Your controller can expose any of these methods:

```python
controller.execute(planned_action_dict)
controller.run_action(planned_action_dict)
controller.send_action(action_vector_or_action_dict)
controller.play_trajectory(trajectory)
controller.replay_trajectory(trajectory)

controller.pause(affected_arm="both_arms")
controller.resume(affected_arm="both_arms")
controller.stop(affected_arm="both_arms")
controller.emergency_stop()
controller.disconnect()
```

You do not need all of them.

For command gating, expose one execution method.

For runtime interruption, expose one stop/pause method.

For OpenARM on current LeRobot `main`, use:

```python
from harness import LeRobotOpenArmController, OpenArmLeRobotAdapter

controller = LeRobotOpenArmController(
    robot=openarm_robot,
    inference_engine=policy_inference_engine,  # optional but recommended
    interpolator=action_interpolator,          # optional
    stop_mode="hold",                         # or "disconnect"
)
robot = OpenArmLeRobotAdapter(controller=controller)
```

`pause()` pauses inference if available, reads `robot.get_observation()`, builds a
hold action from current `.pos` keys, and sends it through `robot.send_action()`.

`resume()` resets the interpolator/inference engine if available, then resumes
inference. The camera watchdog keeps running outside LeRobot.

`stop()` stops inference and either holds the current pose or disconnects,
depending on `stop_mode`.

## 3. Command-Gating Example

```python
import cv2

from harness import OpenArmLeRobotAdapter, WatchdogPerceptionAdapter, safe_execute

camera = cv2.VideoCapture(0)
perception = WatchdogPerceptionAdapter(camera_id="external_webcam_1")
robot = OpenArmLeRobotAdapter(controller=openarm_controller)

decision = safe_execute(
    {
        "id": "act_pick_knife_001",
        "type": "pick",
        "object": "sharp_tool",
        "arm": "left_arm",
        "source": "teleop_replay",
        "payload": {
            "trajectory": replay_trajectory
        },
    },
    camera=camera,
    perception_adapter=perception,
    robot_adapter=robot,
)

print(decision.decision)
```

Expected behavior:

- If a hand is near the sharp tool, `decision.decision == "BLOCK"` and no robot execution method is called.
- If the scene is safe, `decision.decision == "ALLOW"` and the replay/action is sent.

## 4. Runtime Interruption Pattern

Use this when OpenARM is already moving and the external watchdog should pause or stop motion.

The important design point is that the camera loop never stops. When the robot
is paused, perception keeps running. Once the scene is safe for several
consecutive frames, the harness may call `resume`.

```python
import time

from harness import RuntimeWatchdogSupervisor, OpenArmLeRobotAdapter, WatchdogPerceptionAdapter

perception = WatchdogPerceptionAdapter(camera_id="external_webcam_1")
robot = OpenArmLeRobotAdapter(controller=openarm_controller)
supervisor = RuntimeWatchdogSupervisor(
    perception_adapter=perception,
    robot_adapter=robot,
    clear_frames_before_resume=10,
    auto_resume=True,
)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    state = supervisor.step(frame, now=time.time())
    print(state.mode, state.last_decision.decision)

    time.sleep(0.02)
```

`RuntimeWatchdogSupervisor` logs each decision, rate-limits pause calls by state
transition, and resumes only after `clear_frames_before_resume` safe frames.

Recommended semantics:

- `PAUSE`: can auto-resume after the scene is safe.
- `STOP`: should usually be terminal and require operator confirmation before any resume.
- `RESUME`: should go through LeRobot/OpenARM only if your controller supports a safe resume.

## 5. Controller Shim Example

If your LeRobot object does not have the method names expected by the harness, wrap it.

```python
from harness import LeRobotOpenArmController

controller = LeRobotOpenArmController(
    robot=openarm_lerobot_robot,
    inference_engine=ctx.policy.inference,
    interpolator=interpolator,
    stop_mode="hold",
)
```

Then:

```python
robot = OpenArmLeRobotAdapter(controller=controller)
```

If you are inside LeRobot rollout code, the relevant objects are usually:

- `ctx.hardware.robot_wrapper` or `ctx.hardware.robot_wrapper.inner` for the robot.
- `ctx.policy.inference` for the inference engine.
- the strategy's `ActionInterpolator` for interpolated action state.

## 6. Important Safety Notes

- Validate `pause` and `stop` on the real OpenARM setup before presenting runtime interruption.
- Treat both arms as `both_arms` by default.
- Use `BLOCK` only before sending an action.
- Use `PAUSE` or `STOP` after motion has started.
- Keep the VLM out of the critical interruption path.
- Log the scene context, decision, and robot command result for every intervention.

## 7. Integration Checklist

- [ ] Identify the actual OpenARM/LeRobot controller object.
- [ ] Record the exact LeRobot version or commit used by the OpenARM setup.
- [ ] Confirm how to send a scripted action or teleop replay.
- [ ] Confirm whether `pause`, `stop`, `emergency_stop`, or safe hold exists.
- [ ] Create a shim if method names or control semantics differ.
- [ ] Run dry-run gating first.
- [ ] Test `BLOCK`: no action is sent.
- [ ] Test `ALLOW`: the intended action executes.
- [ ] Test `PAUSE` or `STOP`: both arms stop or hold safely.
- [ ] Keep the camera watchdog running while paused.
- [ ] Test `RESUME`: only after several safe frames and, ideally, operator approval.
- [ ] Record JSONL events for dashboard/report.

## 8. Do We Need The LeRobot Version?

For the generic harness architecture: no. The adapter is duck-typed and only
needs a Python object with execution and pause/stop/resume methods.

For a correct real OpenARM integration: yes, it is strongly recommended. LeRobot
APIs, robot wrappers, and action formats can differ across versions or commits.
Record at least:

```bash
python -c "import lerobot; print(getattr(lerobot, '__version__', 'unknown'))"
pip show lerobot
git -C /path/to/lerobot rev-parse HEAD
```

If LeRobot has no installed package version in your setup, keep the git commit
hash for the LeRobot checkout used to control OpenARM.

## 9. Relevant LeRobot Main APIs

Observed on LeRobot `main` commit `3dd19d043e2f3fe5673b13ea0ebe4f31884c0797`:

- `Robot` base class defines `connect`, `get_observation`, `send_action`, and `disconnect`.
- `OpenArmFollower.send_action(action, custom_kp=None, custom_kd=None)` sends `.pos` motor goals after joint-limit clipping.
- `OpenArmFollower.disconnect()` disconnects the CAN bus and cameras.
- `BiOpenArmFollower.send_action(action, ...)` splits `left_` and `right_` prefixed action keys and calls each arm's `send_action`.
- `InferenceEngine` defines optional `pause()` and `resume()` hooks.
- RTC inference implements `pause()` and `resume()` by clearing/setting its active thread event.
- DAgger's pause behavior is a useful reference: pause the engine, keep sending the last action to hold position, then reset/resume the engine when returning to autonomous mode.
