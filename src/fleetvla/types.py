"""Immutable types shared by runtimes, backends, and schedulers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SessionConfig:
    session_id: str
    control_hz: float
    chunk_size: int
    request_threshold_s: float = 0.1
    network_latency_s: float = 0.0
    latency_budget_s: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must not be empty")
        if not _is_finite_number(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size <= 0
        ):
            raise ValueError("chunk_size must be positive")
        timing_values = (
            self.request_threshold_s,
            self.network_latency_s,
            self.latency_budget_s,
        )
        if any(not _is_finite_number(value) or value < 0 for value in timing_values):
            raise ValueError("timing values must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Observation:
    session_id: str
    sequence: int
    generation: int
    captured_at_s: float
    payload: Any = None


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    session_id: str
    generation: int
    ready_sequence: int | None
    request_time_s: float | None
    buffer_steps: int
    buffer_horizon_s: float
    control_hz: float
    latency_budget_s: float
    network_latency_s: float
    connected: bool
    in_flight_sequence: int | None

    @property
    def is_ready(self) -> bool:
        return self.ready_sequence is not None


@dataclass(frozen=True, slots=True)
class FleetSnapshot:
    now_s: float
    sessions: tuple[SessionSnapshot, ...]
    max_batch_size: int

    @property
    def ready_sessions(self) -> tuple[SessionSnapshot, ...]:
        return tuple(session for session in self.sessions if session.is_ready)


@dataclass(frozen=True, slots=True)
class InferenceCostModel:
    base_latency_s: float
    per_item_latency_s: float

    def __post_init__(self) -> None:
        if any(
            not _is_finite_number(value) or value < 0
            for value in (self.base_latency_s, self.per_item_latency_s)
        ):
            raise ValueError("inference latency must be finite and non-negative")

    def estimate(self, batch_size: int) -> float:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.base_latency_s + self.per_item_latency_s * batch_size


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    session_ids: tuple[str, ...]
    reason: str = ""

    def __post_init__(self) -> None:
        if len(set(self.session_ids)) != len(self.session_ids):
            raise ValueError("a schedule decision cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class ActionChunk:
    session_id: str
    observation_sequence: int
    generation: int
    actions: tuple[Any, ...]
    produced_at_s: float
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("an action chunk must contain at least one action")


@dataclass(frozen=True, slots=True)
class ActionCommand:
    session_id: str
    generation: int
    observation_sequence: int
    action_index: int
    value: Any
    observation_captured_at_s: float


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Framework-neutral shape contract for endpoint payloads."""

    name: str
    shape: tuple[int | None, ...]

    def validate(self, payload: Mapping[str, Any]) -> None:
        if self.name not in payload:
            raise ValueError(f"missing payload field: {self.name}")
        actual = _shape_of(payload[self.name])
        if len(actual) != len(self.shape) or any(
            expected is not None and expected != received
            for expected, received in zip(self.shape, actual)
        ):
            raise ValueError(
                f"field {self.name!r} has shape {actual}, expected {self.shape}"
            )


def _shape_of(value: Any) -> tuple[int, ...]:
    declared = getattr(value, "shape", None)
    if declared is not None:
        return tuple(int(size) for size in declared)
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(
            f"cannot establish a numeric shape for {type(value).__name__}"
        )
    if isinstance(value, (int, float, bool)):
        return ()
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shapes = tuple(_shape_of(child) for child in value)
        if len(set(child_shapes)) != 1:
            raise ValueError("payload field is a ragged sequence")
        return (len(value),) + child_shapes[0]
    raise ValueError(
        f"cannot establish a shape for {type(value).__name__}"
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
