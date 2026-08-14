"""Clock implementations used by deterministic and wall-clock runtimes."""

from __future__ import annotations

import math
import time


class VirtualClock:
    """A manually advanced monotonic clock for deterministic simulations."""

    def __init__(self, start_s: float = 0.0) -> None:
        if not isinstance(start_s, (int, float)) or not math.isfinite(start_s):
            raise ValueError("virtual clock start must be finite")
        self._now_s = float(start_s)

    def now(self) -> float:
        return self._now_s

    def advance_to(self, when_s: float) -> None:
        when_s = float(when_s)
        if not math.isfinite(when_s):
            raise ValueError("virtual time must be finite")
        if when_s < self._now_s:
            raise ValueError("virtual time cannot move backwards")
        self._now_s = when_s

    def advance(self, duration_s: float) -> None:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration must be finite and non-negative")
        self._now_s += duration_s


class MonotonicClock:
    """Wall-clock implementation with the same read interface."""

    def now(self) -> float:
        return time.monotonic()
