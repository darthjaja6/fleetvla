"""Structured runtime events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    time_s: float
    kind: str
    session_id: str | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def append(
        self,
        time_s: float,
        kind: str,
        session_id: str | None = None,
        **details: Any,
    ) -> None:
        self._events.append(Event(time_s, kind, session_id, details or None))
