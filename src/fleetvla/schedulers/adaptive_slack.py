"""Batch-aware slack scheduling for action buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import FleetSnapshot, InferenceCostModel, ScheduleDecision, SessionSnapshot
from .base import batch_limit


@dataclass(frozen=True, slots=True)
class AdaptiveSlackConfig:
    batch_size_limit: int | None = None
    transport_margin_s: float = 0.01

    def __post_init__(self) -> None:
        if self.batch_size_limit is not None and self.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if not math.isfinite(self.transport_margin_s) or self.transport_margin_s < 0:
            raise ValueError("transport_margin_s must be finite and non-negative")


class AdaptiveSlackScheduler:
    """Protect the least-slack session, then batch work that remains feasible."""

    config_type = AdaptiveSlackConfig

    def __init__(self, config: AdaptiveSlackConfig | None = None) -> None:
        self.config = config or AdaptiveSlackConfig()

    def _slack(
        self,
        session: SessionSnapshot,
        batch_size: int,
        costs: InferenceCostModel,
    ) -> float:
        return (
            session.buffer_horizon_s
            - costs.estimate(batch_size)
            - session.network_latency_s
            - self.config.transport_margin_s
        )

    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        ready = sorted(
            fleet.ready_sessions,
            key=lambda session: (
                self._slack(session, 1, costs),
                session.request_time_s,
                session.session_id,
            ),
        )
        if not ready:
            return ScheduleDecision((), "no ready sessions")
        limit = min(len(ready), batch_limit(fleet, self.config.batch_size_limit))
        selected_count = 1
        if self._slack(ready[0], 1, costs) > 0:
            for size in range(2, limit + 1):
                if (
                    min(self._slack(session, size, costs) for session in ready[:size])
                    < 0
                ):
                    break
                selected_count = size
        selected = tuple(session.session_id for session in ready[:selected_count])
        return ScheduleDecision(
            selected,
            "protect least slack; batch while selected sessions remain feasible",
        )
