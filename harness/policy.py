"""Deterministic safety policy engine."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.models import Decision, PlannedAction, Policy


class PolicyEngine:
    """Rule-based harness decision layer.

    Perception can be model-based; this layer stays deterministic and auditable.
    """

    def evaluate(
        self,
        planned_action: PlannedAction | dict[str, Any] | None,
        scene_context: dict[str, Any],
        policies: Sequence[Policy | dict[str, Any]],
        evidence_frame_id: str | None = None,
    ) -> Decision:
        action = _coerce_action(planned_action)
        active = []
        for policy in policies:
            coerced = _coerce_policy(policy)
            if coerced.enabled:
                active.append(coerced)

        for policy in active:
            if policy.id == "P1" and action is not None:
                decision = self._human_proximity_around_hazardous_tool(
                    action, scene_context, policy, evidence_frame_id
                )
                if decision:
                    return decision
            if policy.id == "P1W":
                decision = self._blade_tip_aimed_near_hand(
                    action, scene_context, policy, evidence_frame_id
                )
                if decision:
                    return decision

        return Decision(
            decision="ALLOW",
            severity="none",
            rule="no_policy_violation",
            reason="No enabled safety policy was violated.",
            mitigation="Proceed under active monitoring.",
            planned_action_id=action.id if action else None,
            evidence_frame_id=evidence_frame_id,
            affected_arm=action.arm if action else "both_arms",
        )

    def _human_proximity_around_hazardous_tool(
        self,
        action: PlannedAction,
        scene_context: dict[str, Any],
        policy: Policy,
        evidence_frame_id: str | None,
    ) -> Decision | None:
        c = policy.conditions
        action_types = set(c.get("action_types", ["pick", "move", "manipulate"]))
        hazardous_objects = set(c.get("hazardous_objects", ["sharp_tool", "knife", "blade", "needle"]))
        hazards = set(c.get("hazards", ["blade_tip_near_hand", "blade_tip_aimed_at_hand", "sharp_near_person"]))

        scene_hazards = set(scene_context.get("hazards") or [])
        human_present = bool(scene_context.get("zones", {}).get("human_in_workspace"))
        object_is_hazardous = action.object in hazardous_objects
        action_is_relevant = action.type in action_types
        scene_is_unsafe = bool(scene_hazards & hazards) or (
            human_present and "sharp_tool" in scene_context.get("objects", [])
        )

        if action_is_relevant and object_is_hazardous and scene_is_unsafe:
            return Decision(
                decision=policy.decision_on_violation,
                severity=policy.severity,
                rule="human_proximity_around_hazardous_tool",
                reason="A human hand or person is in an unsafe relation to the sharp tool targeted by the planned action.",
                mitigation="Remove the hand from the workspace before retrying the OpenARM action.",
                planned_action_id=action.id,
                evidence_frame_id=evidence_frame_id,
                affected_arm=action.arm,
                policy_id=policy.id,
            )
        return None

    def _blade_tip_aimed_near_hand(
        self,
        action: PlannedAction | None,
        scene_context: dict[str, Any],
        policy: Policy,
        evidence_frame_id: str | None,
    ) -> Decision | None:
        hazards = set(scene_context.get("hazards") or [])
        if "blade_tip_aimed_at_hand" not in hazards:
            return None
        return Decision(
            decision=policy.decision_on_violation,
            severity=policy.severity,
            rule="blade_tip_aimed_near_hand",
            reason="The observed blade tip is close to and pointed toward a human fingertip.",
            mitigation="Pause or stop OpenARM and require the hand to leave the workspace before resume.",
            planned_action_id=action.id if action else None,
            evidence_frame_id=evidence_frame_id,
            affected_arm=action.arm if action else "both_arms",
            policy_id=policy.id,
        )


def default_policies() -> list[Policy]:
    return [
        Policy(
            id="P1",
            name="Human Proximity Around Hazardous Tool",
            enabled=True,
            severity="critical",
            decision_on_violation="BLOCK",
            conditions={
                "action_types": ["pick", "move", "manipulate"],
                "hazardous_objects": ["sharp_tool", "knife", "blade", "needle"],
                "hazards": ["blade_tip_near_hand", "blade_tip_aimed_at_hand", "sharp_near_person"],
            },
        ),
        Policy(
            id="P1W",
            name="Blade Tip Aimed Near Hand",
            enabled=True,
            severity="critical",
            decision_on_violation="PAUSE",
            conditions={"hazards": ["blade_tip_aimed_at_hand"]},
        ),
    ]


def _coerce_action(action: PlannedAction | dict[str, Any] | None) -> PlannedAction | None:
    if action is None:
        return None
    if isinstance(action, PlannedAction):
        return action
    return PlannedAction.from_dict(action)


def _coerce_policy(policy: Policy | dict[str, Any]) -> Policy:
    if isinstance(policy, Policy):
        return policy
    return Policy.from_dict(policy)
