"""Shared schema validation for version-1 artifact events."""

from __future__ import annotations

import math
from typing import Any


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_events(events: list[Any], expected_sessions: set[str]) -> None:
    if not events:
        raise ValueError("artifact events must not be empty")
    previous_time = -math.inf
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("artifact events must contain objects")
        _require_keys(event, "time_s", "kind", "session_id", "details")
        _finite_number(event["time_s"], "event time", minimum=0)
        if event["time_s"] < previous_time:
            raise ValueError("artifact event times must be non-decreasing")
        previous_time = event["time_s"]
        if not isinstance(event["kind"], str) or not event["kind"]:
            raise ValueError("event kind must be a non-empty string")
        if event["session_id"] is not None and not isinstance(event["session_id"], str):
            raise ValueError("event session_id must be a string or null")
        if (
            event["session_id"] is not None
            and event["session_id"] not in expected_sessions
        ):
            raise ValueError("event session_id is not a configured session")
        if event["details"] is not None and not isinstance(event["details"], dict):
            raise ValueError("event details must be an object or null")
        _validate_event_contract(event, expected_sessions)


def _validate_event_contract(
    event: dict[str, Any], expected_sessions: set[str]
) -> None:
    kind = event["kind"]
    global_kinds = {
        "batch_dispatched",
        "dispatch_deferred",
        "inference_started",
        "inference_completed",
        "scheduler_decision",
        "scheduler_failed",
        "scheduler_cost_estimate",
        "inference_failed",
    }
    no_detail_kinds = {
        "session_registered",
        "action_starved",
        "chunk_rejected_stale",
        "chunk_rejected_unexpected",
        "action_ack_rejected_stale",
        "action_ack_rejected_duplicate",
        "action_ack_rejected_unexpected",
        "session_disconnected",
        "observation_dropped_backpressure",
        "observation_suppressed_backpressure",
        "endpoint_observation_unavailable",
        "endpoint_fallback",
        "endpoint_episode_boundary",
    }
    known_kinds = (
        global_kinds
        | no_detail_kinds
        | {
            "observation_ready",
            "request_dispatched",
            "chunk_rejected_horizon",
            "chunk_accepted",
            "action_dequeued",
            "action_sent_endpoint",
            "action_accepted_endpoint",
            "action_executed",
            "action_rejected_endpoint",
            "session_reconnected",
            "session_reset",
            "endpoint_observation_failed",
            "endpoint_fallback_failed",
            "endpoint_action_failed",
            "endpoint_close_failed",
            "control_ticks_missed",
            "endpoint_task_step",
        }
    )
    if kind not in known_kinds:
        raise ValueError(f"unsupported artifact event kind: {kind!r}")
    if (kind in global_kinds) != (event["session_id"] is None):
        raise ValueError(f"event {kind} has an invalid session_id")
    if kind in no_detail_kinds:
        if event["details"] is not None:
            raise ValueError(f"event {kind} must not include details")
        return

    if kind == "batch_dispatched":
        details = _event_details(
            event, "batch_size", "session_ids", "reason", "selected_state"
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        session_ids = details["session_ids"]
        if (
            not isinstance(session_ids, list)
            or len(session_ids) != batch_size
            or any(
                not isinstance(session_id, str) or session_id not in expected_sessions
                for session_id in session_ids
            )
            or len(set(session_ids)) != batch_size
        ):
            raise ValueError("batch_dispatched session_ids are invalid")
        if not isinstance(details["reason"], str):
            raise ValueError("batch reason must be a string")
        selected = details["selected_state"]
        if not isinstance(selected, list) or len(selected) != batch_size:
            raise ValueError("selected_state must match the batch")
        for state, session_id in zip(selected, session_ids):
            if not isinstance(state, dict):
                raise ValueError("selected_state entries must be objects")
            _require_exact_keys(
                state, "session_id", "buffer_horizon_s", "request_age_s"
            )
            if state["session_id"] != session_id:
                raise ValueError("selected_state sessions must match the batch")
            _finite_number(state["buffer_horizon_s"], "buffer_horizon_s", minimum=0)
            _finite_number(state["request_age_s"], "request_age_s", minimum=0)
        return

    if kind == "dispatch_deferred":
        details = _event_details(event, "defer_until_s", "reason")
        _finite_number(details["defer_until_s"], "defer_until_s", minimum=0)
        if details["defer_until_s"] <= event["time_s"]:
            raise ValueError("dispatch deferral must be in the future")
        if not isinstance(details["reason"], str):
            raise ValueError("dispatch deferral reason must be a string")
        return

    if kind == "scheduler_decision":
        details = _event_details(
            event,
            "latency_s",
            "selected_session_ids",
            "deferred",
            "fallback",
        )
        _finite_number(details["latency_s"], "scheduler latency", minimum=0)
        session_ids = details["selected_session_ids"]
        if (
            not isinstance(session_ids, list)
            or any(
                not isinstance(session_id, str) or session_id not in expected_sessions
                for session_id in session_ids
            )
            or len(set(session_ids)) != len(session_ids)
        ):
            raise ValueError("scheduler decision sessions are invalid")
        for field in ("deferred", "fallback"):
            if type(details[field]) is not bool:
                raise ValueError(f"scheduler decision {field} must be boolean")
        if details["deferred"] == bool(session_ids):
            raise ValueError("scheduler decision must select work or defer")
        return

    if kind == "scheduler_failed":
        details = _event_details(event, "error", "fallback")
        _error_string(details["error"])
        if details["fallback"] != "edf":
            raise ValueError("scheduler failure fallback must be edf")
        return

    if kind == "inference_completed":
        details = _event_details(
            event,
            "batch_size",
            optional=(
                "latency_s",
                "backend_reported_latency_s",
                "output_shapes",
                "execution_horizons",
            ),
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        for field in ("latency_s", "backend_reported_latency_s"):
            if field in details:
                _finite_number(details[field], field, minimum=0)
        paired = ("output_shapes" in details, "execution_horizons" in details)
        if paired[0] != paired[1]:
            raise ValueError(
                "output shapes and execution horizons must appear together"
            )
        if paired[0]:
            shapes = details["output_shapes"]
            horizons = details["execution_horizons"]
            if (
                not isinstance(shapes, list)
                or len(shapes) != batch_size
                or not isinstance(horizons, list)
                or len(horizons) != batch_size
            ):
                raise ValueError("inference output metadata must match batch_size")
            for shape in shapes:
                if not isinstance(shape, list) or not shape:
                    raise ValueError("output shapes must be non-empty arrays")
                for size in shape:
                    _positive_integer(size, "output dimension")
                if shape[0] != batch_size:
                    raise ValueError("output batch dimension must match batch_size")
            for horizon in horizons:
                _positive_integer(horizon, "execution horizon")
        return

    if kind == "inference_started":
        details = _event_details(
            event, "batch_size", optional=("latency_s", "scheduler_reason")
        )
        _positive_integer(details["batch_size"], "batch_size")
        if "latency_s" in details:
            _finite_number(details["latency_s"], "latency_s", minimum=0)
        if "scheduler_reason" in details and not isinstance(
            details["scheduler_reason"], str
        ):
            raise ValueError("scheduler_reason must be a string")
        return

    if kind in {"scheduler_cost_estimate"}:
        details = _event_details(
            event,
            "batch_size",
            "base_latency_s",
            "per_item_latency_s",
            "estimated_latency_s",
            optional=("batch_latency_s",),
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        for field in ("base_latency_s", "per_item_latency_s", "estimated_latency_s"):
            _finite_number(details[field], field, minimum=0)
        profile = details.get("batch_latency_s", [])
        if not isinstance(profile, list):
            raise ValueError("batch latency profile must be an array")
        for value in profile:
            _finite_number(value, "batch latency", minimum=0)
        if profile and batch_size > len(profile):
            raise ValueError("batch latency profile does not cover batch_size")
        expected = (
            profile[batch_size - 1]
            if profile
            else details["base_latency_s"] + details["per_item_latency_s"] * batch_size
        )
        if not math.isclose(details["estimated_latency_s"], expected):
            raise ValueError("estimated latency does not match the cost model")
        return

    if kind == "inference_failed":
        details = _event_details(event, "error", "session_ids", "phase")
        _error_string(details["error"])
        session_ids = details["session_ids"]
        if (
            not isinstance(session_ids, list)
            or not session_ids
            or any(
                not isinstance(session_id, str) or session_id not in expected_sessions
                for session_id in session_ids
            )
            or len(set(session_ids)) != len(session_ids)
        ):
            raise ValueError("inference_failed session_ids are invalid")
        if details["phase"] not in {"prepare", "inference"}:
            raise ValueError("inference failure phase is invalid")
        return

    details = _event_details(event, *_EVENT_REQUIRED_DETAILS[kind])
    if kind == "observation_ready":
        _non_negative_integer(details["sequence"], "sequence")
    elif kind == "request_dispatched":
        _non_negative_integer(details["sequence"], "sequence")
        _positive_integer(details["batch_size"], "batch_size")
    elif kind == "chunk_rejected_horizon":
        _positive_integer(details["expected_actions"], "expected_actions")
        _positive_integer(details["received_actions"], "received_actions")
        if details["expected_actions"] == details["received_actions"]:
            raise ValueError("horizon rejection requires unequal action counts")
    elif kind == "chunk_accepted":
        _non_negative_integer(details["sequence"], "sequence")
        _positive_integer(details["actions"], "actions")
        _finite_number(details["action_age_s"], "action_age_s", minimum=0)
    elif kind in {
        "action_dequeued",
        "action_sent_endpoint",
        "action_accepted_endpoint",
        "action_executed",
        "action_rejected_endpoint",
    }:
        _non_negative_integer(details["sequence"], "sequence")
        _non_negative_integer(details["action_index"], "action_index")
        if kind == "action_dequeued":
            _finite_number(details["action_age_s"], "action_age_s", minimum=0)
            _non_negative_integer(details["remaining_steps"], "remaining_steps")
        elif kind == "action_sent_endpoint":
            _finite_number(details["deadline_s"], "deadline_s", minimum=0)
        elif kind in {"action_executed", "action_rejected_endpoint"}:
            _finite_number(details["action_age_s"], "action_age_s", minimum=0)
    elif kind in {"session_reconnected", "session_reset"}:
        _non_negative_integer(details["generation"], "generation")
    elif kind in {
        "endpoint_observation_failed",
        "endpoint_fallback_failed",
        "endpoint_action_failed",
        "endpoint_close_failed",
    }:
        _error_string(details["error"])
    elif kind == "control_ticks_missed":
        _positive_integer(details["count"], "missed tick count")
    elif kind == "endpoint_task_step":
        _finite_number(details["reward"], "task reward")
        for field in ("success", "terminated", "truncated"):
            if type(details[field]) is not bool:
                raise ValueError(f"{field} must be boolean")


_EVENT_REQUIRED_DETAILS = {
    "observation_ready": ("sequence",),
    "request_dispatched": ("sequence", "batch_size"),
    "chunk_rejected_horizon": ("expected_actions", "received_actions"),
    "chunk_accepted": ("sequence", "actions", "action_age_s"),
    "action_dequeued": (
        "remaining_steps",
        "sequence",
        "action_index",
        "action_age_s",
    ),
    "action_sent_endpoint": ("sequence", "action_index", "deadline_s"),
    "action_accepted_endpoint": ("sequence", "action_index"),
    "action_executed": ("sequence", "action_index", "action_age_s"),
    "action_rejected_endpoint": ("sequence", "action_index", "action_age_s"),
    "session_reconnected": ("generation",),
    "session_reset": ("generation",),
    "endpoint_observation_failed": ("error",),
    "endpoint_fallback_failed": ("error",),
    "endpoint_action_failed": ("error",),
    "endpoint_close_failed": ("error",),
    "control_ticks_missed": ("count",),
    "endpoint_task_step": ("reward", "success", "terminated", "truncated"),
}


def _require_keys(values: dict[str, Any], *keys: str) -> None:
    missing = set(keys) - set(values)
    if missing:
        raise ValueError(f"artifact is missing fields: {', '.join(sorted(missing))}")


def _event_details(
    event: dict[str, Any],
    *required: str,
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    details = event["details"]
    if not isinstance(details, dict):
        raise ValueError(f"event {event['kind']} must include details")
    _require_keys(details, *required)
    unknown = set(details) - set(required) - set(optional)
    if unknown:
        raise ValueError(
            f"event {event['kind']} has unknown details: {', '.join(sorted(unknown))}"
        )
    return details


def _require_exact_keys(values: dict[str, Any], *keys: str) -> None:
    _require_keys(values, *keys)
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError(f"unexpected fields: {', '.join(sorted(unknown))}")


def _positive_integer_value(value: Any, name: str) -> int:
    _positive_integer(value, name)
    return value


def _error_string(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("event error must be a non-empty string")


def _require_strings(values: dict[str, Any], *keys: str) -> None:
    _require_keys(values, *keys)
    for key in keys:
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"{key} must be a non-empty string")


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (minimum is not None and (value <= minimum if strict else value < minimum))
    ):
        qualifier = "positive" if strict else "finite"
        raise ValueError(f"{name} must be {qualifier}")


def _non_negative_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _unit_interval(value: Any, name: str) -> None:
    _finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
