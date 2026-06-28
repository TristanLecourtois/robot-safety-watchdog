"""Safety harness layer around the vision watchdog."""
from harness.core import safe_execute
from harness.logger import JsonlAuditLogger
from harness.models import Decision, PlannedAction, Policy, RobotCommandResult
from harness.policy import PolicyEngine, default_policies
from harness.robot import LeRobotOpenArmController, OpenArmLeRobotAdapter, RobotAdapter
from harness.runtime import RuntimeWatchdogSupervisor, run_runtime_watchdog

__all__ = [
    "Decision",
    "JsonlAuditLogger",
    "LeRobotOpenArmController",
    "OpenArmLeRobotAdapter",
    "PlannedAction",
    "Policy",
    "PolicyEngine",
    "RobotAdapter",
    "RobotCommandResult",
    "RuntimeWatchdogSupervisor",
    "WatchdogPerceptionAdapter",
    "default_policies",
    "run_runtime_watchdog",
    "safe_execute",
]


def __getattr__(name):
    if name == "WatchdogPerceptionAdapter":
        from harness.perception import WatchdogPerceptionAdapter

        return WatchdogPerceptionAdapter
    raise AttributeError(name)
