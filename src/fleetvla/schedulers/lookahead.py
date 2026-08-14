"""One-step closed-loop lookahead for heterogeneous action-chunk fleets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import (
    FleetSnapshot,
    InferenceCostModel,
    ScheduleDecision,
    SessionSnapshot,
)
from .base import batch_limit


@dataclass(frozen=True, slots=True)
class LookaheadConfig:
    """Configuration for the paper's deployed one-epoch Lookahead variant."""

    batch_size_limit: int | None = None
    evaluation_horizon_s: float = 1.0

    def __post_init__(self) -> None:
        if self.batch_size_limit is not None and self.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive")
        if (
            not math.isfinite(self.evaluation_horizon_s)
            or self.evaluation_horizon_s <= 0
        ):
            raise ValueError("evaluation_horizon_s must be finite and positive")


class LookaheadScheduler:
    """Choose the batch with most weighted executed time per inference second.

    This is a FleetVLA adaptation of Lookahead at depth L=1 from *Action Chunk
    Scheduling for Batched Robot Policy Serving*. It evaluates the current
    buffer, predicted chunk arrival, per-session execution horizon and service
    weight over a fixed future window, then dispatches the first (and only)
    batch from the highest-scoring schedule.
    """

    config_type = LookaheadConfig

    def __init__(self, config: LookaheadConfig | None = None) -> None:
        self.config = config or LookaheadConfig()

    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        ready = fleet.ready_sessions
        if not ready:
            return ScheduleDecision((), "no ready sessions")
        limit = min(len(ready), batch_limit(fleet, self.config.batch_size_limit))
        best_batch: tuple[str, ...] | None = None
        best_score = -math.inf
        for batch in _candidate_batches(fleet, limit):
            latency_s = costs.estimate(len(batch))
            reward = sum(
                session.service_weight
                * _new_chunk_executed_time(
                    session,
                    selected=session.session_id in batch,
                    inference_latency_s=latency_s,
                    horizon_s=self.config.evaluation_horizon_s,
                )
                for session in fleet.sessions
                if session.connected
            )
            score = reward if latency_s == 0 else reward / latency_s
            if score > best_score:
                best_batch = batch
                best_score = score
        if best_batch is None:
            raise RuntimeError("lookahead found no candidate batch")
        return ScheduleDecision(
            best_batch,
            "maximize weighted executed time per inference second over "
            f"{self.config.evaluation_horizon_s:g}s",
        )


def _candidate_batches(fleet: FleetSnapshot, limit: int) -> tuple[tuple[str, ...], ...]:
    """Enumerate priority-tier prefixes in EDF order, matching Armory pruning."""

    ordered = sorted(
        fleet.ready_sessions,
        key=lambda session: (
            fleet.now_s + session.buffer_horizon_s,
            session.request_time_s,
            session.session_id,
        ),
    )
    tiers: dict[float, list[str]] = {}
    for session in ordered:
        tiers.setdefault(session.service_weight, []).append(session.session_id)
    tier_ids = [tiers[weight] for weight in sorted(tiers, reverse=True)]
    candidates: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for size in range(1, limit + 1):
        for counts in _bounded_compositions(
            size, tuple(len(tier) for tier in tier_ids)
        ):
            batch = tuple(
                session_id
                for tier, count in zip(tier_ids, counts)
                for session_id in tier[:count]
            )
            if batch not in seen:
                candidates.append(batch)
                seen.add(batch)
    return tuple(candidates)


def _bounded_compositions(total: int, capacities: tuple[int, ...]):
    if len(capacities) == 1:
        if total <= capacities[0]:
            yield (total,)
        return
    for count in range(min(total, capacities[0]) + 1):
        for remainder in _bounded_compositions(total - count, capacities[1:]):
            yield (count,) + remainder


def _new_chunk_executed_time(
    session: SessionSnapshot,
    *,
    selected: bool,
    inference_latency_s: float,
    horizon_s: float,
) -> float:
    if not selected:
        return 0.0
    buffered_s = session.buffer_steps / session.control_hz
    chunk_s = session.chunk_size / session.control_hz
    arrival_s = inference_latency_s + session.network_latency_s
    chunk_start_s = max(buffered_s, arrival_s)
    return min(chunk_s, max(0.0, horizon_s - chunk_start_s))
