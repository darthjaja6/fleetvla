"""First-in, first-out scheduling."""

from __future__ import annotations

from ..types import FleetSnapshot, InferenceCostModel, ScheduleDecision
from .base import BatchConfig, batch_limit

FIFOConfig = BatchConfig


class FIFOScheduler:
    config_type = FIFOConfig

    def __init__(self, config: FIFOConfig | None = None) -> None:
        self.config = config or FIFOConfig()

    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        del costs
        ready = sorted(
            fleet.ready_sessions,
            key=lambda session: (session.request_time_s, session.session_id),
        )
        selected = tuple(
            session.session_id
            for session in ready[: batch_limit(fleet, self.config.batch_size_limit)]
        )
        return ScheduleDecision(selected, "oldest observations first")
