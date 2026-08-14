"""Action-chunk boundary for stateful VLA/WAM-style predictors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..backend import BackendResult
from ..types import ActionChunk, InferenceCostModel, Observation


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    actions: tuple[Any, ...]
    next_state: Any = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    observations: tuple[Observation, ...]
    states: tuple[Any, ...]
    tokens: tuple[int, ...]


class StatefulPolicyBackend:
    """Keep opaque per-session policy state outside scheduler snapshots."""

    def __init__(
        self,
        predict_batch: Callable[
            [tuple[Any, ...], tuple[Any, ...]], Sequence[PolicyPrediction]
        ],
        *,
        base_latency_s: float,
        per_item_latency_s: float,
    ) -> None:
        self.predict_batch = predict_batch
        self.cost_model = InferenceCostModel(base_latency_s, per_item_latency_s)
        self._state: dict[str, Any] = {}
        self._pending: dict[tuple[str, int, int], tuple[int, Any]] = {}
        self._tokens: dict[str, int] = {}
        self._lock = Lock()

    def prepare_batch(self, observations: Sequence[Observation]) -> _PreparedBatch:
        observations = tuple(observations)
        if not observations:
            raise ValueError("cannot infer an empty batch")
        with self._lock:
            tokens = tuple(
                self._tokens.get(observation.session_id, 0)
                for observation in observations
            )
            states = tuple(
                self._state.get(observation.session_id) for observation in observations
            )
        return _PreparedBatch(observations, states, tokens)

    def infer_prepared(
        self, prepared: _PreparedBatch, started_at_s: float
    ) -> BackendResult:
        observations = prepared.observations
        predictions = tuple(
            self.predict_batch(
                tuple(observation.payload for observation in observations),
                prepared.states,
            )
        )
        if len(predictions) != len(observations):
            raise ValueError("predict_batch returned the wrong batch size")
        latency_s = self.cost_model.estimate(len(observations))
        chunks = []
        for observation, prediction, token in zip(
            observations, predictions, prepared.tokens
        ):
            if not isinstance(prediction, PolicyPrediction):
                raise TypeError("predict_batch must return PolicyPrediction values")
            with self._lock:
                if self._tokens.get(observation.session_id, 0) == token:
                    key = (
                        observation.session_id,
                        observation.generation,
                        observation.sequence,
                    )
                    self._pending[key] = (token, prediction.next_state)
            chunks.append(
                ActionChunk(
                    session_id=observation.session_id,
                    observation_sequence=observation.sequence,
                    generation=observation.generation,
                    actions=prediction.actions,
                    produced_at_s=started_at_s + latency_s,
                    auxiliary=prediction.auxiliary,
                    action_index_start=observation.action_index_start,
                )
            )
        return BackendResult(latency_s, tuple(chunks))

    def infer(
        self, observations: Sequence[Observation], started_at_s: float
    ) -> BackendResult:
        return self.infer_prepared(self.prepare_batch(observations), started_at_s)

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._state.pop(session_id, None)
            self._pending = {
                key: value
                for key, value in self._pending.items()
                if key[0] != session_id
            }
            self._tokens[session_id] = self._tokens.get(session_id, 0) + 1

    def commit_chunk(self, chunk: ActionChunk, accepted: bool) -> None:
        key = (chunk.session_id, chunk.generation, chunk.observation_sequence)
        with self._lock:
            pending = self._pending.pop(key, None)
            if pending is None or not accepted:
                return
            token, next_state = pending
            if self._tokens.get(chunk.session_id, 0) == token:
                self._state[chunk.session_id] = next_state

    def state_for_testing(self, session_id: str) -> Any:
        with self._lock:
            return self._state.get(session_id)
