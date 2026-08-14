"""Fair rotation among ready sessions."""

from __future__ import annotations

from ..types import FleetSnapshot, InferenceCostModel, ScheduleDecision
from .base import BatchConfig, batch_limit

RoundRobinConfig = BatchConfig


class RoundRobinScheduler:
    config_type = RoundRobinConfig

    def __init__(self, config: RoundRobinConfig | None = None) -> None:
        self.config = config or RoundRobinConfig()
        self._next_session_id: str | None = None

    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        del costs
        ready = sorted(session.session_id for session in fleet.ready_sessions)
        if not ready:
            return ScheduleDecision((), "no ready sessions")
        start = 0
        if self._next_session_id is not None:
            start = next(
                (
                    index
                    for index, session_id in enumerate(ready)
                    if session_id >= self._next_session_id
                ),
                0,
            )
        rotated = ready[start:] + ready[:start]
        selected = rotated[: batch_limit(fleet, self.config.batch_size_limit)]
        last_index = ready.index(selected[-1])
        self._next_session_id = ready[(last_index + 1) % len(ready)]
        return ScheduleDecision(tuple(selected), "rotate across ready sessions")
