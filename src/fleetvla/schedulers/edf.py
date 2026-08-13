"""Earliest-deadline-first scheduling."""

from __future__ import annotations

from ..types import FleetSnapshot, InferenceCostModel, ScheduleDecision, SessionSnapshot
from .base import BatchConfig, batch_limit


EDFConfig = BatchConfig


def _deadline(now_s: float, session: SessionSnapshot) -> float:
    request_time_s = (
        session.request_time_s if session.request_time_s is not None else now_s
    )
    request_deadline = request_time_s + session.latency_budget_s
    buffer_deadline = now_s + session.buffer_horizon_s
    return min(request_deadline, buffer_deadline)


class EDFScheduler:
    config_type = EDFConfig

    def __init__(self, config: EDFConfig | None = None) -> None:
        self.config = config or EDFConfig()

    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        del costs
        ready = sorted(
            fleet.ready_sessions,
            key=lambda session: (
                _deadline(fleet.now_s, session),
                session.request_time_s,
                session.session_id,
            ),
        )
        selected = tuple(
            session.session_id
            for session in ready[: batch_limit(fleet, self.config.batch_size_limit)]
        )
        return ScheduleDecision(selected, "earliest buffer or request deadline")
