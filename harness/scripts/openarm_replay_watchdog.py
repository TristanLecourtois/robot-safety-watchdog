"""Replay a LeRobot episode while the harness pauses on camera danger.

The replay loop is owned by this script, not by ``lerobot-replay``. That lets
the runtime watchdog pause the robot mid-episode, keep holding the current pose,
and resume the same replay index after the danger clears.
"""
from __future__ import annotations

import argparse
import importlib
import threading
import time
from pathlib import Path
from typing import Any, Callable

from harness import (
    JsonlAuditLogger,
    LeRobotOpenArmController,
    OpenArmLeRobotAdapter,
    RuntimeWatchdogSupervisor,
)
from harness.models import Decision


class RememberingPerceptionAdapter:
    """Keep the last scene context so visualizers can inspect watchdog output."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.last_scene_context: dict[str, Any] | None = None

    def parse_frame(self, frame, now: float | None = None) -> dict[str, Any]:
        self.last_scene_context = self.adapter.parse_frame(frame, now=now)
        return self.last_scene_context


class RerunWatchdogVisualizer:
    """Optional Rerun visualization for camera frames and watchdog detections."""

    def __init__(self, *, app_id: str, mode: str, save_path: str | None = None):
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError(
                "Rerun is not installed in this Python environment. Install it with: "
                "python3 -m pip install rerun-sdk"
            ) from exc

        self.rr = rr
        rr.init(app_id)
        if mode == "spawn":
            rr.spawn()
        elif mode == "connect":
            rr.connect()
        elif mode == "save":
            if not save_path:
                raise ValueError("--rerun-save is required when --rerun-mode save")
            rr.save(save_path)
        else:
            raise ValueError("--rerun-mode must be spawn, connect, or save")

    def log(self, *, frame, frame_i: int, scene_context: dict[str, Any] | None, state) -> None:
        import cv2

        rr = self.rr
        rr.set_time_sequence("watchdog_frame", frame_i)
        rr.log("camera/image", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if not scene_context:
            return

        detections = scene_context.get("detections") or []
        boxes = [d["box"] for d in detections if d.get("box")]
        labels = [
            f"{d.get('object') or d.get('label')} {float(d.get('confidence', 0.0)):.2f}"
            for d in detections
            if d.get("box")
        ]
        if boxes:
            rr.log("camera/detections", self._boxes2d(boxes, labels))

        hazards = scene_context.get("hazards") or []
        decision = state.last_decision.decision if state.last_decision else "NONE"
        rule = state.last_decision.rule if state.last_decision else "-"
        rr.log("watchdog/mode", rr.TextLog(state.mode))
        rr.log("watchdog/decision", rr.TextLog(f"{decision} | {rule}"))
        rr.log("watchdog/hazards", rr.TextLog(", ".join(hazards) if hazards else "none"))

    def _boxes2d(self, boxes, labels):
        rr = self.rr
        try:
            return rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels)
        except AttributeError:
            return rr.Boxes2D(array=boxes, array_format="xyxy", labels=labels)


class LockedController:
    """Serialize robot access between replay and watchdog threads."""

    def __init__(self, controller: LeRobotOpenArmController):
        self.controller = controller
        self.lock = threading.RLock()

    @property
    def paused(self) -> bool:
        return bool(getattr(self.controller, "_paused", False))

    def execute_hold_once(self) -> None:
        with self.lock:
            self.controller._send_hold_once()

    def get_observation(self):
        with self.lock:
            return self.controller.robot.get_observation()

    def send_action(self, action):
        with self.lock:
            return self.controller.robot.send_action(action)

    def pause(self, affected_arm: str = "both_arms") -> None:
        with self.lock:
            return self.controller.pause(affected_arm=affected_arm)

    def resume(self, affected_arm: str = "both_arms") -> None:
        with self.lock:
            return self.controller.resume(affected_arm=affected_arm)

    def stop(self, affected_arm: str = "both_arms") -> None:
        with self.lock:
            return self.controller.stop(affected_arm=affected_arm)

    def disconnect(self) -> None:
        with self.lock:
            return self.controller.robot.disconnect()


class AnyHazardPausePolicyEngine:
    """Pause on any watchdog rule hazard, then allow resume when clear."""

    def evaluate(self, planned_action, scene_context, policies, evidence_frame_id=None):
        hazards = scene_context.get("hazards") or []
        severity = scene_context.get("max_rule_severity")
        if hazards or severity in {"warning", "critical"}:
            return Decision(
                decision="PAUSE",
                severity=severity or "warning",
                rule="any_watchdog_hazard",
                reason=f"Watchdog detected runtime danger: {', '.join(hazards) or severity}.",
                mitigation="Hold OpenARM until fresh camera frames are clear.",
                evidence_frame_id=evidence_frame_id,
                affected_arm="both_arms",
                policy_id="demo_any_hazard_pause",
            )
        return Decision(
            decision="ALLOW",
            severity="none",
            rule="no_watchdog_hazard",
            reason="No watchdog hazard detected.",
            mitigation="Continue replay under active monitoring.",
            evidence_frame_id=evidence_frame_id,
            affected_arm="both_arms",
            policy_id="demo_any_hazard_pause",
        )


class PresencePausePolicyEngine:
    """Pause when a hand/person is detected, useful for validating vision-triggered hold."""

    def __init__(self, *, include_person: bool):
        self.include_person = include_person

    def evaluate(self, planned_action, scene_context, policies, evidence_frame_id=None):
        objects = set(scene_context.get("objects") or [])
        hands = scene_context.get("hands") or []
        detections = scene_context.get("detections") or []

        hand_seen = bool(hands) or "human_hand" in objects or any(
            d.get("object") == "human_hand" for d in detections
        )
        person_seen = "person" in objects or any(d.get("object") == "person" for d in detections)
        unsafe = hand_seen or (self.include_person and person_seen)

        if unsafe:
            rule = "human_presence" if self.include_person else "hand_presence"
            return Decision(
                decision="PAUSE",
                severity="warning",
                rule=rule,
                reason="Watchdog detected a human hand/person in the camera view."
                if self.include_person
                else "Watchdog detected a human hand in the camera view.",
                mitigation="Hold OpenARM until the human leaves the camera view.",
                evidence_frame_id=evidence_frame_id,
                affected_arm="both_arms",
                policy_id=f"demo_{rule}",
            )

        return Decision(
            decision="ALLOW",
            severity="none",
            rule="no_human_presence" if self.include_person else "no_hand_presence",
            reason="No hand/person detected in the camera view."
            if self.include_person
            else "No hand detected in the camera view.",
            mitigation="Continue replay under active monitoring.",
            evidence_frame_id=evidence_frame_id,
            affected_arm="both_arms",
            policy_id="demo_presence_pause",
        )


class SharpHandAssociationPausePolicyEngine:
    """Pause only when a sharp object is near a detected hand/person.

    This intentionally does not pause on a sharp object alone, or on a hand
    alone. It is the runtime replay policy for knife + hand proximity demos.
    """

    ASSOCIATION_HAZARDS = {
        "blade_tip_near_hand",
        "blade_tip_aimed_at_hand",
        "sharp_near_person",
    }

    def __init__(self, proximity_px: float):
        self.proximity_px = proximity_px

    def evaluate(self, planned_action, scene_context, policies, evidence_frame_id=None):
        hazards = set(scene_context.get("hazards") or [])
        matched_hazards = sorted(hazards & self.ASSOCIATION_HAZARDS)
        matched_by_boxes = self._sharp_hand_box_proximity(scene_context)

        if matched_hazards or matched_by_boxes:
            reason = ", ".join(matched_hazards) if matched_hazards else "sharp_tool_box_near_hand"
            return Decision(
                decision="PAUSE",
                severity="critical" if "blade_tip_aimed_at_hand" in hazards else "high",
                rule="sharp_hand_association",
                reason=f"Sharp object and human hand/person are associated in the scene: {reason}.",
                mitigation="Hold OpenARM until the hand leaves the sharp object's danger zone.",
                evidence_frame_id=evidence_frame_id,
                affected_arm="both_arms",
                policy_id="sharp_hand_association_pause",
            )

        return Decision(
            decision="ALLOW",
            severity="none",
            rule="no_sharp_hand_association",
            reason="No close association between a sharp object and a hand/person.",
            mitigation="Continue replay under active monitoring.",
            evidence_frame_id=evidence_frame_id,
            affected_arm="both_arms",
            policy_id="sharp_hand_association_pause",
        )

    def _sharp_hand_box_proximity(self, scene_context) -> bool:
        detections = scene_context.get("detections") or []
        sharps = [d for d in detections if d.get("object") == "sharp_tool"]
        humans = [d for d in detections if d.get("object") in {"human_hand", "person"}]
        if not sharps or not humans:
            return False
        return any(_box_distance(a.get("box"), b.get("box")) < self.proximity_px for a in sharps for b in humans)


def _box_distance(a, b) -> float:
    """Distance between two xyxy boxes in pixels; 0 when they overlap."""
    if not a or not b:
        return float("inf")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx * dx + dy * dy) ** 0.5


def _load_factory(factory_ref: str) -> Callable[[], Any]:
    if ":" not in factory_ref:
        raise ValueError("--robot-factory must look like module:function")
    module_name, function_name = factory_ref.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{factory_ref} is not callable")
    return factory


def _maybe_connect(robot: Any, calibrate: bool) -> None:
    if getattr(robot, "is_connected", False):
        return
    connect = getattr(robot, "connect", None)
    if not callable(connect):
        return
    try:
        connect(calibrate=calibrate)
    except TypeError:
        connect()


def _build_watchdog_config(args: argparse.Namespace):
    import config as watchdog_config

    cfg = watchdog_config.CONFIG
    cfg.camera_index = _camera_source(args.camera_index)
    cfg.frame_width = args.frame_width
    cfg.frame_height = args.frame_height
    cfg.detector_backend = args.detector_backend
    cfg.yolo_model = args.yolo_model
    cfg.yoloe_model = args.yoloe_model
    if args.min_confidence is not None:
        cfg.thresholds.min_confidence = args.min_confidence
    cfg.use_hand_landmarks = not args.no_hand_landmarks
    cfg.enable_world_model = False
    if args.disable_vlm:
        cfg.vlm_model = ""
        cfg.vlm_interval_s = 10**9
        cfg.vlm_min_interval_s = 10**9
    return cfg


def _watchdog_loop(
    *,
    args: argparse.Namespace,
    locked_controller: LockedController,
    stop_event: threading.Event,
    ready_event: threading.Event,
) -> None:
    import cv2

    from harness import WatchdogPerceptionAdapter

    camera_source = _camera_source(args.camera_index)
    print(f"watchdog camera: opening {camera_source!r}")
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.camera_buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_source!r}")

    adapter = OpenArmLeRobotAdapter(controller=locked_controller)
    perception_adapter = RememberingPerceptionAdapter(
        WatchdogPerceptionAdapter(
            camera_id=f"camera_{args.camera_index}",
        )
    )
    rerun_viz = (
        RerunWatchdogVisualizer(
            app_id=args.rerun_app_id,
            mode=args.rerun_mode,
            save_path=args.rerun_save,
        )
        if args.rerun
        else None
    )
    if args.pause_on == "any-hazard":
        policy_engine = AnyHazardPausePolicyEngine()
    elif args.pause_on == "hand-presence":
        policy_engine = PresencePausePolicyEngine(include_person=False)
    elif args.pause_on == "human-presence":
        policy_engine = PresencePausePolicyEngine(include_person=True)
    elif args.pause_on == "sharp-hand":
        policy_engine = SharpHandAssociationPausePolicyEngine(proximity_px=args.sharp_hand_proximity_px)
    else:
        policy_engine = None
    supervisor = RuntimeWatchdogSupervisor(
        perception_adapter=perception_adapter,
        robot_adapter=adapter,
        policy_engine=policy_engine,
        logger=JsonlAuditLogger(Path(args.log)),
        clear_frames_before_resume=args.clear_frames_before_resume,
        unsafe_frames_before_pause=args.unsafe_frames_before_pause,
        auto_resume=True,
        stop_is_terminal=False,
    )

    frame_i = 0
    try:
        while not stop_event.is_set():
            ok, frame = _read_latest_frame(cap, args.drop_stale_camera_frames)
            if not ok:
                time.sleep(0.02)
                continue
            state = supervisor.step(frame, now=time.time())
            if frame_i + 1 >= args.watchdog_warmup_frames:
                ready_event.set()
            if rerun_viz is not None:
                rerun_viz.log(
                    frame=frame,
                    frame_i=frame_i,
                    scene_context=perception_adapter.last_scene_context,
                    state=state,
                )
            if args.print_watchdog_every and frame_i % args.print_watchdog_every == 0:
                decision = state.last_decision.decision if state.last_decision else "NONE"
                rule = state.last_decision.rule if state.last_decision else "-"
                scene = perception_adapter.last_scene_context or {}
                detections = scene.get("detections") or []
                objects = ",".join(scene.get("objects") or []) or "none"
                det_text = ", ".join(
                    f"{d.get('object') or d.get('label')}:{float(d.get('confidence', 0.0)):.2f}"
                    for d in detections[:6]
                ) or "none"
                print(
                    f"watchdog: mode={state.mode} decision={decision} rule={rule} "
                    f"objects={objects} detections={det_text}"
                )
            frame_i += 1
            time.sleep(max(args.watchdog_sleep_s, 0.0))
    finally:
        cap.release()


def _load_episode_actions(args: argparse.Namespace):
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import ACTION

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode])
    actions = dataset.select_columns(ACTION)
    action_names = dataset.features[ACTION]["names"]
    fps = args.fps or dataset.fps
    return dataset, actions, action_names, fps


def _camera_source(value: str):
    return int(value) if value.isdigit() else value


def _read_latest_frame(cap, drop_stale_frames: int):
    ok, frame = cap.read()
    if not ok:
        return ok, frame
    for _ in range(max(drop_stale_frames, 0)):
        if not cap.grab():
            break
        grabbed, latest = cap.retrieve()
        if grabbed:
            frame = latest
    return True, frame


def run(args: argparse.Namespace) -> None:
    from lerobot.processor import make_default_robot_action_processor
    from lerobot.utils.robot_utils import precise_sleep

    robot = _load_factory(args.robot_factory)()
    if args.connect:
        print(f"connect: calling robot.connect(calibrate={args.calibrate}) if available")
        _maybe_connect(robot, calibrate=args.calibrate)

    controller = LeRobotOpenArmController(
        robot=robot,
        inference_engine=None,
        interpolator=None,
        hold_hz=args.hold_hz,
        stop_mode=args.stop_mode,
    )
    locked_controller = LockedController(controller)

    dataset, actions, action_names, fps = _load_episode_actions(args)
    robot_action_processor = make_default_robot_action_processor()

    stop_event = threading.Event()
    ready_event = threading.Event()
    watchdog_thread = threading.Thread(
        target=_watchdog_loop,
        kwargs={
            "args": args,
            "locked_controller": locked_controller,
            "stop_event": stop_event,
            "ready_event": ready_event,
        },
        daemon=True,
    )
    watchdog_thread.start()
    print(
        "watchdog: waiting for detector/camera warmup "
        f"({args.watchdog_warmup_frames} processed frame(s))"
    )
    if not ready_event.wait(timeout=args.watchdog_ready_timeout_s):
        stop_event.set()
        watchdog_thread.join(timeout=2.0)
        raise TimeoutError(
            "Watchdog did not process a camera frame before timeout. "
            "Check camera path, YOLO model download/load, and detector dependencies."
        )
    print("watchdog: ready; starting replay")

    print(
        f"replay: episode={args.episode} frames={dataset.num_frames} fps={fps} "
        f"pause_on={args.pause_on}"
    )
    idx = 0
    try:
        while idx < dataset.num_frames:
            start_t = time.perf_counter()

            if locked_controller.paused:
                locked_controller.execute_hold_once()
                precise_sleep(max(1 / fps - (time.perf_counter() - start_t), 0.0))
                continue

            action_array = actions[idx]["action"]
            action = {name: action_array[i] for i, name in enumerate(action_names)}
            robot_obs = locked_controller.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            locked_controller.send_action(processed_action)
            idx += 1

            if args.print_replay_every and idx % args.print_replay_every == 0:
                print(f"replay: sent frame {idx}/{dataset.num_frames}")

            precise_sleep(max(1 / fps - (time.perf_counter() - start_t), 0.0))
    finally:
        stop_event.set()
        watchdog_thread.join(timeout=2.0)
        if args.stop_at_end:
            locked_controller.stop()
        if args.disconnect:
            locked_controller.disconnect()

    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay LeRobot episode and pause on watchdog danger.")
    parser.add_argument("--robot-factory", required=True, help="Python factory returning openarm_robot")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--disconnect", action="store_true")
    parser.add_argument("--stop-at-end", action="store_true")
    parser.add_argument("--stop-mode", choices=("hold", "disconnect"), default="hold")
    parser.add_argument("--hold-hz", type=float, default=20.0)
    parser.add_argument(
        "--camera-index",
        default="0",
        help="OpenCV camera index or device path, e.g. 0 or /dev/video4",
    )
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--detector-backend", choices=("yoloe", "yolo"), default="yoloe")
    parser.add_argument("--yolo-model", default="yolo11l-seg.pt")
    parser.add_argument("--yoloe-model", default="yoloe-11s-seg.pt")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--drop-stale-camera-frames", type=int, default=0)
    parser.add_argument("--no-hand-landmarks", action="store_true")
    parser.add_argument("--disable-vlm", action="store_true", default=True)
    parser.add_argument("--watchdog-sleep-s", type=float, default=0.0)
    parser.add_argument("--watchdog-warmup-frames", type=int, default=1)
    parser.add_argument("--watchdog-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--clear-frames-before-resume", type=int, default=10)
    parser.add_argument("--unsafe-frames-before-pause", type=int, default=1)
    parser.add_argument(
        "--pause-on",
        choices=("policy", "any-hazard", "sharp-hand", "hand-presence", "human-presence"),
        default="sharp-hand",
    )
    parser.add_argument(
        "--sharp-hand-proximity-px",
        type=float,
        default=120.0,
        help="box-level fallback distance for --pause-on sharp-hand",
    )
    parser.add_argument("--log", default="openarm_replay_watchdog.jsonl")
    parser.add_argument("--print-watchdog-every", type=int, default=15)
    parser.add_argument("--print-replay-every", type=int, default=30)
    parser.add_argument("--rerun", action="store_true", help="visualize camera, YOLO boxes, and decisions in Rerun")
    parser.add_argument("--rerun-app-id", default="openarm_replay_watchdog")
    parser.add_argument("--rerun-mode", choices=("spawn", "connect", "save"), default="spawn")
    parser.add_argument("--rerun-save", default=None, help="path to .rrd when --rerun-mode save is used")
    args = parser.parse_args()

    _build_watchdog_config(args)
    try:
        run(args)
    except KeyboardInterrupt:
        raise SystemExit("\ninterrupted by operator\n")


if __name__ == "__main__":
    main()
