"""Inference backend contracts and deterministic synthetic backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .types import ActionChunk, InferenceCostModel, Observation


@dataclass(frozen=True, slots=True)
class BackendResult:
    latency_s: float
    chunks: tuple[ActionChunk, ...]


class SyntheticBackend:
    """Produces fixed-size action chunks with batch-dependent latency."""

    def __init__(
        self,
        *,
        chunk_size: int = 4,
        chunk_sizes: Mapping[str, int] | None = None,
        base_latency_s: float = 0.02,
        per_item_latency_s: float = 0.005,
        action_factory: Callable[[Observation, int], Any] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.chunk_sizes = dict(chunk_sizes or {})
        if any(size <= 0 for size in self.chunk_sizes.values()):
            raise ValueError("chunk sizes must be positive")
        self.cost_model = InferenceCostModel(base_latency_s, per_item_latency_s)
        self._action_factory = action_factory or self._default_action

    @staticmethod
    def _default_action(observation: Observation, index: int) -> dict[str, Any]:
        return {"observation": observation.sequence, "step": index}

    def infer(self, observations: Iterable[Observation], started_at_s: float) -> BackendResult:
        batch = tuple(observations)
        if not batch:
            raise ValueError("cannot infer an empty batch")
        latency_s = self.cost_model.estimate(len(batch))
        produced_at_s = started_at_s + latency_s
        chunks = tuple(
            ActionChunk(
                session_id=observation.session_id,
                observation_sequence=observation.sequence,
                generation=observation.generation,
                actions=tuple(
                    self._action_factory(observation, index)
                    for index in range(
                        self.chunk_sizes.get(observation.session_id, self.chunk_size)
                    )
                ),
                produced_at_s=produced_at_s,
            )
            for observation in batch
        )
        return BackendResult(latency_s, chunks)
