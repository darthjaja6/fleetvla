"""Metric schema and event-reconciliation validation."""

from __future__ import annotations

from typing import Any

from .artifact_schema import (
    _finite_number,
    _non_negative_integer,
    _require_keys,
    _unit_interval,
)
from .benchmark import compute_metrics
from .simulation import RobotSpec, SimulationResult
from .trace import Event


def _validate_benchmark_metrics(metrics: Any, expected_sessions: set[str]) -> None:
    if not isinstance(metrics, dict):
        raise ValueError("benchmark metrics must be an object")
    _require_keys(
        metrics,
        "useful_actions",
        "starvation_frequency",
        "starvation_duration_s",
        "action_age_p50_s",
        "action_age_p95_s",
        "fairness",
        "batch_sizes",
        "backend_utilization",
        "per_session",
    )
    _non_negative_integer(metrics["useful_actions"], "useful_actions")
    has_delivery_metrics = "sent_actions" in metrics or "accepted_actions" in metrics
    if has_delivery_metrics:
        if not {"sent_actions", "accepted_actions"} <= metrics.keys():
            raise ValueError("action delivery metrics must be provided together")
        for field in ("sent_actions", "accepted_actions"):
            _non_negative_integer(metrics[field], field)
        if (
            not metrics["useful_actions"]
            <= metrics["accepted_actions"]
            <= metrics["sent_actions"]
        ):
            raise ValueError("action delivery metrics must be monotonically decreasing")
    _unit_interval(metrics["starvation_frequency"], "starvation_frequency")
    _finite_number(metrics["starvation_duration_s"], "starvation_duration_s", minimum=0)
    for field in ("action_age_p50_s", "action_age_p95_s"):
        if metrics[field] is not None:
            _finite_number(metrics[field], field, minimum=0)
    if (
        metrics["action_age_p50_s"] is not None
        and metrics["action_age_p95_s"] is not None
        and metrics["action_age_p95_s"] < metrics["action_age_p50_s"]
    ):
        raise ValueError("action_age_p95_s cannot be below action_age_p50_s")
    _unit_interval(metrics["fairness"], "fairness")
    _unit_interval(metrics["backend_utilization"], "backend_utilization")
    batches = metrics["batch_sizes"]
    if not isinstance(batches, list) or any(
        type(size) is not int or size <= 0 for size in batches
    ):
        raise ValueError("batch_sizes must contain positive integers")
    per_session = metrics["per_session"]
    if not isinstance(per_session, dict) or set(per_session) != expected_sessions:
        raise ValueError("per_session metrics must match configured sessions")
    for session_metrics in per_session.values():
        if not isinstance(session_metrics, dict):
            raise ValueError("per_session metrics must be objects")
        _require_keys(
            session_metrics,
            "actions",
            "starved_ticks",
            "starvation_duration_s",
            "useful_progress_ratio",
        )
        _non_negative_integer(session_metrics["actions"], "session actions")
        if has_delivery_metrics:
            if not {"sent_actions", "accepted_actions"} <= session_metrics.keys():
                raise ValueError(
                    "session action delivery metrics must be provided together"
                )
            for field in ("sent_actions", "accepted_actions"):
                _non_negative_integer(session_metrics[field], f"session {field}")
            if (
                not session_metrics["actions"]
                <= session_metrics["accepted_actions"]
                <= session_metrics["sent_actions"]
            ):
                raise ValueError(
                    "session action delivery metrics must be monotonically decreasing"
                )
        _non_negative_integer(session_metrics["starved_ticks"], "session starved_ticks")
        _finite_number(
            session_metrics["starvation_duration_s"],
            "session starvation_duration_s",
            minimum=0,
        )
        _unit_interval(
            session_metrics["useful_progress_ratio"], "useful_progress_ratio"
        )


def _validate_metrics_match_events(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    robots: tuple[RobotSpec, ...],
    duration_s: float,
) -> None:
    try:
        event_objects = tuple(Event(**event) for event in events)
        calculated = compute_metrics(
            SimulationResult(duration_s, event_objects), robots
        ).as_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact event details are invalid") from error
    if "sent_actions" not in metrics:
        calculated.pop("sent_actions")
        calculated.pop("accepted_actions")
        for session_metrics in calculated["per_session"].values():
            session_metrics.pop("sent_actions")
            session_metrics.pop("accepted_actions")
    if calculated != metrics:
        raise ValueError("artifact metrics do not match its events")
