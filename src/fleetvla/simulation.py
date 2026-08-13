"""Deterministic discrete-event fleet simulator."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from .backend import BackendResult, SyntheticBackend
from .clock import VirtualClock
from .runtime import FleetRuntime
from .schedulers import FIFOScheduler, Scheduler
from .trace import Event
from .types import ActionChunk, SessionConfig


@dataclass(frozen=True, slots=True)
class RobotSpec:
    session_id: str
    control_hz: float
    chunk_size: int = 4
    request_threshold_s: float = 0.1
    network_latency_s: float = 0.0
    latency_budget_s: float = 0.25
    start_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_delay_s) or self.start_delay_s < 0:
            raise ValueError("start_delay_s must be finite and non-negative")

    def session_config(self) -> SessionConfig:
        return SessionConfig(
            session_id=self.session_id,
            control_hz=self.control_hz,
            chunk_size=self.chunk_size,
            request_threshold_s=self.request_threshold_s,
            network_latency_s=self.network_latency_s,
            latency_budget_s=self.latency_budget_s,
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    duration_s: float
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("simulation duration must be finite and positive")

    def count(self, kind: str, session_id: str | None = None) -> int:
        return sum(
            event.kind == kind
            and (session_id is None or event.session_id == session_id)
            for event in self.events
        )


class FleetSimulator:
    """Runs control ticks concurrently with batched inference events."""

    def __init__(
        self,
        robots: list[RobotSpec],
        *,
        backend: SyntheticBackend | None = None,
        scheduler: Scheduler | None = None,
        max_batch_size: int = 8,
        observation_schedule: tuple[tuple[float, str], ...] | None = None,
    ) -> None:
        if not robots:
            raise ValueError("at least one robot is required")
        if len({robot.session_id for robot in robots}) != len(robots):
            raise ValueError("robot session_id values must be unique")
        self.robots = {robot.session_id: robot for robot in robots}
        self.clock = VirtualClock()
        self.runtime = FleetRuntime(self.clock, max_batch_size=max_batch_size)
        for robot in robots:
            self.runtime.register(robot.session_config())
        self.backend = backend or SyntheticBackend(
            chunk_sizes={robot.session_id: robot.chunk_size for robot in robots}
        )
        self.scheduler = scheduler or FIFOScheduler()
        self._events: list[tuple[float, int, int, str, Any]] = []
        self._next_order = 0
        self._backend_busy = False
        self._dispatch_wakeup_s: float | None = None
        self._deferred_ready: tuple[tuple[str, int | None], ...] | None = None
        self._has_run = False
        self._observation_schedule = observation_schedule
        if observation_schedule is not None:
            for time_s, session_id in observation_schedule:
                if not math.isfinite(time_s) or time_s < 0:
                    raise ValueError(
                        "observation times must be finite and non-negative"
                    )
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError(
                        "observation session IDs must be non-empty strings"
                    )

    def run(self, duration_s: float) -> SimulationResult:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self._has_run:
            raise RuntimeError("a FleetSimulator instance can only be run once")
        self._has_run = True
        if self._observation_schedule is None:
            observations = tuple(
                (robot.start_delay_s, robot.session_id)
                for robot in self.robots.values()
            )
        else:
            observations = self._observation_schedule
        for time_s, session_id in observations:
            if session_id not in self.robots:
                raise ValueError(f"trace references unknown session: {session_id}")
            self._push(time_s, "observe", session_id)
        for robot in self.robots.values():
            self._push(
                robot.start_delay_s + 1.0 / robot.control_hz,
                "control_tick",
                (robot.session_id, 1),
            )

        while self._events and self._events[0][0] <= duration_s:
            now_s = self._events[0][0]
            self.clock.advance_to(now_s)
            # Process to a fixed point: handlers may enqueue higher-priority work
            # (for example a zero-delay delivery) at this same timestamp.
            while self._events and self._events[0][0] == now_s:
                _, _, _, kind, payload = heapq.heappop(self._events)
                self._handle(kind, payload)
            self._maybe_dispatch()

        self.clock.advance_to(duration_s)
        return SimulationResult(duration_s, self.runtime.events.events)

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "observe":
            session_id = payload
            if not self.runtime.has_pending_request(session_id):
                self.runtime.observe(session_id, {"time_s": self.clock.now()})
            else:
                self.runtime.events.append(
                    self.clock.now(), "observation_dropped_backpressure", session_id
                )
            return
        if kind == "control_tick":
            session_id, tick = payload
            robot = self.robots[session_id]
            self.runtime.consume_action(session_id)
            session = next(
                item
                for item in self.runtime.snapshot().sessions
                if item.session_id == session_id
            )
            if (
                self._observation_schedule is None
                and session.buffer_horizon_s <= robot.request_threshold_s
                and not self.runtime.has_pending_request(session_id)
            ):
                self.runtime.observe(session_id, {"time_s": self.clock.now()})
            self._push(
                robot.start_delay_s + (tick + 1) / robot.control_hz,
                "control_tick",
                (session_id, tick + 1),
            )
            return
        if kind == "inference_complete":
            result: BackendResult = payload
            self._backend_busy = False
            self.runtime.events.append(
                self.clock.now(),
                "inference_completed",
                batch_size=len(result.chunks),
            )
            for chunk in result.chunks:
                delay_s = self.robots[chunk.session_id].network_latency_s
                self._push(self.clock.now() + delay_s, "deliver", chunk)
            return
        if kind == "deliver":
            chunk: ActionChunk = payload
            accepted = self.runtime.accept(chunk)
            commit_chunk = getattr(self.backend, "commit_chunk", None)
            if commit_chunk is not None:
                commit_chunk(chunk, accepted)
            return
        if kind == "dispatch_wakeup":
            if payload == self._dispatch_wakeup_s:
                self._dispatch_wakeup_s = None
                self._deferred_ready = None
            return
        raise AssertionError(f"unknown simulation event: {kind}")

    def _maybe_dispatch(self) -> None:
        if self._backend_busy:
            return
        snapshot = self.runtime.snapshot()
        if not snapshot.ready_sessions:
            self._dispatch_wakeup_s = None
            self._deferred_ready = None
            return
        ready = tuple(
            (session.session_id, session.ready_sequence)
            for session in snapshot.ready_sessions
        )
        if (
            self._dispatch_wakeup_s is not None
            and self.clock.now() < self._dispatch_wakeup_s
            and ready == self._deferred_ready
        ):
            return
        decision = self.scheduler.schedule(snapshot, self.backend.cost_model)
        if decision.defer_until_s is not None:
            if decision.defer_until_s <= self.clock.now():
                raise ValueError("scheduler deferral must be in the future")
            self._dispatch_wakeup_s = decision.defer_until_s
            self._deferred_ready = ready
            self.runtime.events.append(
                self.clock.now(),
                "dispatch_deferred",
                defer_until_s=decision.defer_until_s,
                reason=decision.reason,
            )
            self._push(
                decision.defer_until_s,
                "dispatch_wakeup",
                decision.defer_until_s,
            )
            return
        self._dispatch_wakeup_s = None
        self._deferred_ready = None
        batch = self.runtime.prepare_batch(decision)
        result = self.backend.infer(batch, self.clock.now())
        self.runtime.events.append(
            self.clock.now(),
            "inference_started",
            batch_size=len(batch),
            latency_s=result.latency_s,
            scheduler_reason=decision.reason,
        )
        self._backend_busy = True
        self._push(self.clock.now() + result.latency_s, "inference_complete", result)

    def _push(self, time_s: float, kind: str, payload: Any) -> None:
        priority = {
            "inference_complete": 0,
            "deliver": 1,
            "observe": 2,
            "control_tick": 3,
            "dispatch_wakeup": 4,
        }[kind]
        heapq.heappush(
            self._events,
            (time_s, priority, self._next_order, kind, payload),
        )
        self._next_order += 1
