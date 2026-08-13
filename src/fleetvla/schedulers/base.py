"""Scheduler protocol and shared configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..types import FleetSnapshot, InferenceCostModel, ScheduleDecision


class Scheduler(Protocol):
    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision: ...


@dataclass(frozen=True, slots=True)
class BatchConfig:
    batch_size_limit: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size_limit is not None and self.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")


def batch_limit(fleet: FleetSnapshot, configured: int | None) -> int:
    if configured is None:
        return fleet.max_batch_size
    return min(fleet.max_batch_size, configured)
