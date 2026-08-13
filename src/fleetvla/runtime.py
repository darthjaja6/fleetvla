"""Stateful session lifecycle shared by simulated and physical endpoints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from .trace import EventLog
from .types import (
    ActionChunk,
    ActionCommand,
    FleetSnapshot,
    Observation,
    ScheduleDecision,
    SessionConfig,
    SessionSnapshot,
)


class Clock(Protocol):
    def now(self) -> float: ...


@dataclass(slots=True)
class _Session:
    config: SessionConfig
    generation: int = 0
    next_sequence: int = 0
    ready: Observation | None = None
    in_flight: Observation | None = None
    actions: deque["_BufferedAction"] = field(default_factory=deque)
    connected: bool = True
    acknowledged: set[tuple[int, int, int]] = field(default_factory=set)
    outstanding: set[tuple[int, int, int]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _BufferedAction:
    value: Any
    observation_sequence: int
    observation_captured_at_s: float
    action_index: int
    generation: int


class FleetRuntime:
    """Owns lifecycle state; schedulers only receive immutable snapshots."""

    def __init__(self, clock: Clock, *, max_batch_size: int = 8) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.clock = clock
        self.max_batch_size = max_batch_size
        self.events = EventLog()
        self._sessions: dict[str, _Session] = {}

    def register(self, config: SessionConfig) -> None:
        if config.session_id in self._sessions:
            raise ValueError(f"session already registered: {config.session_id}")
        self._sessions[config.session_id] = _Session(config)
        self.events.append(self.clock.now(), "session_registered", config.session_id)

    def observe(self, session_id: str, payload: Any = None) -> Observation:
        session = self._session(session_id)
        if not session.connected:
            raise RuntimeError(f"session is disconnected: {session_id}")
        if session.ready is not None or session.in_flight is not None:
            raise RuntimeError(f"session already has a pending request: {session_id}")
        observation = Observation(
            session_id=session_id,
            sequence=session.next_sequence,
            generation=session.generation,
            captured_at_s=self.clock.now(),
            payload=payload,
        )
        session.next_sequence += 1
        session.ready = observation
        self.events.append(
            self.clock.now(), "observation_ready", session_id, sequence=observation.sequence
        )
        return observation

    def snapshot(self) -> FleetSnapshot:
        now_s = self.clock.now()
        sessions = tuple(
            SessionSnapshot(
                session_id=session.config.session_id,
                generation=session.generation,
                ready_sequence=session.ready.sequence if session.ready else None,
                request_time_s=session.ready.captured_at_s if session.ready else None,
                buffer_steps=len(session.actions),
                buffer_horizon_s=len(session.actions) / session.config.control_hz,
                control_hz=session.config.control_hz,
                latency_budget_s=session.config.latency_budget_s,
                network_latency_s=session.config.network_latency_s,
                connected=session.connected,
                in_flight_sequence=(
                    session.in_flight.sequence if session.in_flight else None
                ),
            )
            for session in self._sessions.values()
        )
        return FleetSnapshot(now_s, sessions, self.max_batch_size)

    def prepare_batch(self, decision: ScheduleDecision) -> tuple[Observation, ...]:
        if decision.defer_until_s is not None:
            raise ValueError("cannot prepare a deferred schedule decision")
        if not decision.session_ids:
            raise ValueError("scheduler returned an empty batch while work was available")
        if len(decision.session_ids) > self.max_batch_size:
            raise ValueError("scheduler exceeded max_batch_size")
        observations: list[Observation] = []
        dispatch_state: list[dict[str, Any]] = []
        for session_id in decision.session_ids:
            session = self._session(session_id)
            if session.ready is None:
                raise ValueError(f"session is not ready: {session_id}")
            observations.append(session.ready)
            dispatch_state.append(
                {
                    "session_id": session_id,
                    "buffer_horizon_s": len(session.actions)
                    / session.config.control_hz,
                    "request_age_s": self.clock.now()
                    - session.ready.captured_at_s,
                }
            )
        for observation in observations:
            session = self._sessions[observation.session_id]
            session.ready = None
            session.in_flight = observation
            self.events.append(
                self.clock.now(),
                "request_dispatched",
                observation.session_id,
                sequence=observation.sequence,
                batch_size=len(observations),
            )
        self.events.append(
            self.clock.now(),
            "batch_dispatched",
            session_ids=tuple(observation.session_id for observation in observations),
            batch_size=len(observations),
            reason=decision.reason,
            selected_state=dispatch_state,
        )
        return tuple(observations)

    def accept(self, chunk: ActionChunk) -> bool:
        session = self._session(chunk.session_id)
        if not session.connected or session.generation != chunk.generation:
            self.events.append(
                self.clock.now(), "chunk_rejected_stale", chunk.session_id
            )
            return False
        if (
            session.in_flight is None
            or session.in_flight.generation != chunk.generation
            or session.in_flight.sequence != chunk.observation_sequence
        ):
            self.events.append(
                self.clock.now(), "chunk_rejected_unexpected", chunk.session_id
            )
            return False
        if len(chunk.actions) != session.config.chunk_size:
            session.in_flight = None
            self.events.append(
                self.clock.now(),
                "chunk_rejected_horizon",
                chunk.session_id,
                expected_actions=session.config.chunk_size,
                received_actions=len(chunk.actions),
            )
            return False
        observation = session.in_flight
        session.in_flight = None
        session.actions.extend(
            _BufferedAction(
                value=action,
                observation_sequence=chunk.observation_sequence,
                observation_captured_at_s=observation.captured_at_s,
                action_index=index,
                generation=chunk.generation,
            )
            for index, action in enumerate(chunk.actions)
        )
        self.events.append(
            self.clock.now(),
            "chunk_accepted",
            chunk.session_id,
            sequence=chunk.observation_sequence,
            actions=len(chunk.actions),
            action_age_s=self.clock.now() - chunk.produced_at_s,
        )
        return True

    def consume_action(self, session_id: str) -> Any | None:
        command = self.dequeue_action(session_id)
        if command is None:
            return None
        self.acknowledge(command, accepted=True)
        return command.value

    def dequeue_action(self, session_id: str) -> ActionCommand | None:
        session = self._session(session_id)
        if not session.actions:
            self.events.append(self.clock.now(), "action_starved", session_id)
            return None
        action = session.actions.popleft()
        self.events.append(
            self.clock.now(),
            "action_dequeued",
            session_id,
            remaining_steps=len(session.actions),
            sequence=action.observation_sequence,
            action_index=action.action_index,
            action_age_s=self.clock.now() - action.observation_captured_at_s,
        )
        session.outstanding.add(
            (action.generation, action.observation_sequence, action.action_index)
        )
        return ActionCommand(
            session_id=session_id,
            generation=action.generation,
            observation_sequence=action.observation_sequence,
            action_index=action.action_index,
            value=action.value,
            observation_captured_at_s=action.observation_captured_at_s,
        )

    def acknowledge(self, command: ActionCommand, *, accepted: bool) -> bool:
        session = self._session(command.session_id)
        key = (
            command.generation,
            command.observation_sequence,
            command.action_index,
        )
        if command.generation != session.generation:
            self.events.append(
                self.clock.now(), "action_ack_rejected_stale", command.session_id
            )
            return False
        if key in session.acknowledged:
            self.events.append(
                self.clock.now(), "action_ack_rejected_duplicate", command.session_id
            )
            return False
        if key not in session.outstanding:
            self.events.append(
                self.clock.now(), "action_ack_rejected_unexpected", command.session_id
            )
            return False
        session.outstanding.remove(key)
        session.acknowledged.add(key)
        self.events.append(
            self.clock.now(),
            "action_executed" if accepted else "action_rejected_endpoint",
            command.session_id,
            sequence=command.observation_sequence,
            action_index=command.action_index,
            action_age_s=self.clock.now() - command.observation_captured_at_s,
        )
        return True

    def has_pending_request(self, session_id: str) -> bool:
        session = self._session(session_id)
        return session.ready is not None or session.in_flight is not None

    def disconnect(self, session_id: str) -> None:
        session = self._session(session_id)
        session.connected = False
        session.generation += 1
        session.ready = None
        session.in_flight = None
        session.actions.clear()
        session.acknowledged.clear()
        session.outstanding.clear()
        self.events.append(self.clock.now(), "session_disconnected", session_id)

    def reconnect(self, session_id: str) -> None:
        session = self._session(session_id)
        if session.connected:
            raise RuntimeError(f"session is already connected: {session_id}")
        session.connected = True
        self.events.append(
            self.clock.now(),
            "session_reconnected",
            session_id,
            generation=session.generation,
        )

    def reset(self, session_id: str) -> None:
        session = self._session(session_id)
        session.generation += 1
        session.ready = None
        session.in_flight = None
        session.actions.clear()
        session.acknowledged.clear()
        session.outstanding.clear()
        self.events.append(
            self.clock.now(), "session_reset", session_id, generation=session.generation
        )

    def _session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"unknown session: {session_id}") from error
