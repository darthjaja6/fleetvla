"""Public scheduler conformance checks used by contributors and CI."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..types import (
    FleetSnapshot,
    InferenceCostModel,
    ScheduleDecision,
    SessionSnapshot,
)
from .base import Scheduler

_SERVING_DECISION_BUDGET_S = 0.01


class SchedulerConformanceError(ValueError):
    """A scheduler does not satisfy the public decision contract."""


def _snapshot(
    session_ids: tuple[str, ...],
    *,
    now_s: float = 0.1,
    action_execution: str = "sequential-buffer",
) -> FleetSnapshot:
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
            chunk_size=index + 1,
            service_weight=2.0 if index == 0 else 1.0,
        )
        for index, session_id in enumerate(session_ids)
    )
    return FleetSnapshot(
        now_s, sessions, max_batch_size=2, action_execution=action_execution
    )


def _mixed_snapshot() -> FleetSnapshot:
    def session(
        session_id: str,
        *,
        ready_sequence: int | None = None,
        connected: bool = True,
        in_flight_sequence: int | None = None,
    ) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=session_id,
            generation=0,
            ready_sequence=ready_sequence,
            request_time_s=0.1 if ready_sequence is not None else None,
            buffer_steps=0,
            buffer_horizon_s=0,
            control_hz=20,
            latency_budget_s=0.2,
            network_latency_s=0.005,
            connected=connected,
            in_flight_sequence=in_flight_sequence,
        )

    return FleetSnapshot(
        0.15,
        (
            session("ready", ready_sequence=0),
            session("idle"),
            session("in-flight", in_flight_sequence=1),
            session("disconnected", connected=False),
        ),
        max_batch_size=2,
    )


def _priority_stress_snapshot() -> FleetSnapshot:
    sessions = tuple(
        SessionSnapshot(
            session_id=f"priority-{index}",
            generation=0,
            ready_sequence=index,
            request_time_s=index * 0.001,
            buffer_steps=index % 5,
            buffer_horizon_s=(index % 5) / 20,
            control_hz=20,
            latency_budget_s=0.2,
            network_latency_s=0.005,
            connected=True,
            in_flight_sequence=None,
            chunk_size=6 + index % 5,
            service_weight=float(20 - index),
        )
        for index in range(20)
    )
    return FleetSnapshot(0.1, sessions, max_batch_size=8)


def _validate_decision(decision: object, fleet: FleetSnapshot) -> ScheduleDecision:
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
    return decision


def _schedule_checked(
    scheduler: Scheduler,
    fleet: FleetSnapshot,
    costs: InferenceCostModel,
    fixture: str,
) -> ScheduleDecision:
    try:
        decision = scheduler.schedule(fleet, costs)
        return _validate_decision(decision, fleet)
    except Exception as error:
        raise SchedulerConformanceError(f"{fixture}: {error}") from error


async def _check_wall_clock(
    scheduler: Scheduler, fleet: FleetSnapshot, costs: InferenceCostModel
) -> None:
    from ..scheduler_execution import SchedulerExecutionError, SchedulerRunner

    runner = SchedulerRunner(scheduler)
    try:
        try:
            result = await runner.decide(
                fleet, costs, timeout_s=_SERVING_DECISION_BUDGET_S
            )
        except SchedulerExecutionError as error:
            raise SchedulerConformanceError(str(error)) from error
        _validate_decision(result.decision, fleet)
    finally:
        runner.close()


def check_scheduler(factory: Callable[[], Scheduler]) -> tuple[str, ...]:
    """Return passed contract checks or raise with the failing requirement."""

    fixtures = (
        ("three-session sequential fixture", _snapshot(("arm-a", "arm-b", "arm-c"))),
        (
            "changed ready-set latest-indexed fixture",
            _snapshot(
                ("robot-x", "robot-y"),
                now_s=0.12,
                action_execution="latest-indexed",
            ),
        ),
        ("mixed lifecycle fixture", _mixed_snapshot()),
        ("20-tier serving-budget fixture", _priority_stress_snapshot()),
    )
    costs = InferenceCostModel(
        0,
        0,
        (0.025, 0.04, 0.052, 0.064, 0.076, 0.088, 0.1, 0.112),
    )
    first = factory()
    repeated = factory()
    for fixture, fleet in fixtures:
        decision = _schedule_checked(first, fleet, costs, fixture)
        repeated_decision = _schedule_checked(repeated, fleet, costs, fixture)
        if repeated_decision != decision:
            raise SchedulerConformanceError(
                f"{fixture}: scheduler instances are not deterministic "
                "across sequential calls"
            )
    reference = fixtures[0][1]
    empty = FleetSnapshot(reference.now_s, (), reference.max_batch_size)
    try:
        empty_decision = factory().schedule(empty, costs)
    except Exception as error:
        raise SchedulerConformanceError(f"empty-fleet fixture: {error}") from error
    if not isinstance(empty_decision, ScheduleDecision):
        raise SchedulerConformanceError(
            "empty-fleet fixture: scheduler must return a ScheduleDecision"
        )
    if empty_decision.session_ids or empty_decision.defer_until_s is not None:
        raise SchedulerConformanceError(
            "empty-fleet fixture: scheduler selected or deferred work "
            "from an empty fleet"
        )
    try:
        asyncio.run(_check_wall_clock(factory(), fixtures[-1][1], costs))
    except SchedulerConformanceError as error:
        raise SchedulerConformanceError(
            f"20-tier serving-budget fixture: {error}"
        ) from error
    return (
        "selection or future deferral for ready work",
        "ScheduleDecision return type",
        "unique sessions",
        "ready-session subset",
        "batch-size limit",
        "deterministic sequential state",
        "ready-set change handling",
        "10 ms serving decision budget",
        "empty-fleet behavior",
    )
