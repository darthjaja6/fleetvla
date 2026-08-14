"""Public scheduler conformance checks used by contributors and CI."""

from __future__ import annotations

from collections.abc import Callable

from ..types import (
    FleetSnapshot,
    InferenceCostModel,
    ScheduleDecision,
    SessionSnapshot,
)
from .base import Scheduler


class SchedulerConformanceError(ValueError):
    """A scheduler does not satisfy the public decision contract."""


def _snapshot(session_ids: tuple[str, ...]) -> FleetSnapshot:
    sessions = tuple(
        SessionSnapshot(
            session_id=session_id,
            generation=0,
            ready_sequence=index,
            request_time_s=index * 0.01,
            buffer_steps=index,
            buffer_horizon_s=index * 0.05,
            control_hz=20,
            latency_budget_s=0.2,
            network_latency_s=0.005,
            connected=True,
            in_flight_sequence=None,
        )
        for index, session_id in enumerate(session_ids)
    )
    return FleetSnapshot(0.1, sessions, max_batch_size=2)


def check_scheduler(factory: Callable[[], Scheduler]) -> tuple[str, ...]:
    """Return passed contract checks or raise with the failing requirement."""

    fleets = (
        _snapshot(("arm-a", "arm-b", "arm-c")),
        _snapshot(("robot-x", "robot-y")),
    )
    costs = InferenceCostModel(0.02, 0.005)
    for fleet in fleets:
        decision = factory().schedule(fleet, costs)
        if not isinstance(decision, ScheduleDecision):
            raise SchedulerConformanceError("scheduler must return a ScheduleDecision")
        if not isinstance(decision.reason, str):
            raise SchedulerConformanceError("schedule decision reason must be a string")
        if len(set(decision.session_ids)) != len(decision.session_ids):
            raise SchedulerConformanceError(
                "scheduler decision contains duplicate sessions"
            )
        if not decision.session_ids and decision.defer_until_s is None:
            raise SchedulerConformanceError(
                "scheduler must select work or defer to a future time"
            )
        if decision.defer_until_s is not None and decision.defer_until_s <= fleet.now_s:
            raise SchedulerConformanceError(
                "scheduler deferral must be later than fleet.now_s"
            )
        if len(decision.session_ids) > fleet.max_batch_size:
            raise SchedulerConformanceError("scheduler exceeded max_batch_size")
        ready = {session.session_id for session in fleet.ready_sessions}
        if not set(decision.session_ids) <= ready:
            raise SchedulerConformanceError(
                "scheduler selected a session that is not ready"
            )
        repeated = factory().schedule(fleet, costs)
        if repeated != decision:
            raise SchedulerConformanceError(
                "fresh scheduler instances are not deterministic"
            )
    empty = FleetSnapshot(fleets[0].now_s, (), fleets[0].max_batch_size)
    empty_decision = factory().schedule(empty, costs)
    if empty_decision.session_ids or empty_decision.defer_until_s is not None:
        raise SchedulerConformanceError(
            "scheduler selected or deferred work from an empty fleet"
        )
    return (
        "selection or future deferral for ready work",
        "ScheduleDecision return type",
        "unique sessions",
        "ready-session subset",
        "batch-size limit",
        "deterministic fresh instance",
        "empty-fleet behavior",
    )
