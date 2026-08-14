"""Bounded wall-clock execution for synchronous scheduler plugins."""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
from dataclasses import dataclass

from .schedulers import Scheduler
from .types import FleetSnapshot, InferenceCostModel, ScheduleDecision


class SchedulerExecutionError(RuntimeError):
    """A scheduler decision failed or exceeded its wall-clock budget."""


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    decision: ScheduleDecision
    latency_s: float


class SchedulerRunner:
    """Run one stateful scheduler on a daemon worker with bounded responses.

    The worker keeps plugin computation off the serving event loop. If a call
    times out, its eventual result is abandoned and no further work is sent to
    that worker. Python cannot safely kill a thread, so this is timing and
    failure isolation for trusted plugins, not a security sandbox.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        self._requests: queue.Queue[object] = queue.Queue()
        self._responses: queue.Queue[tuple[int, object]] = queue.Queue()
        self._ids = itertools.count()
        self._enabled = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"fleetvla-scheduler-{type(scheduler).__name__}",
            daemon=True,
        )
        self._thread.start()

    async def decide(
        self,
        fleet: FleetSnapshot,
        costs: InferenceCostModel,
        timeout_s: float,
    ) -> SchedulerResult:
        if not self._enabled:
            raise SchedulerExecutionError("scheduler worker is unavailable")
        request_id = next(self._ids)
        started_s = asyncio.get_running_loop().time()
        self._requests.put((request_id, fleet, costs))
        deadline_s = started_s + timeout_s
        while True:
            try:
                response_id, response = self._responses.get_nowait()
            except queue.Empty:
                remaining_s = deadline_s - asyncio.get_running_loop().time()
                if remaining_s <= 0:
                    self._enabled = False
                    raise SchedulerExecutionError(
                        f"scheduler exceeded {timeout_s:.6g}s decision budget"
                    )
                await asyncio.sleep(min(0.0005, remaining_s))
                continue
            if response_id != request_id:
                continue
            if isinstance(response, _SchedulerFailure):
                self._enabled = False
                raise SchedulerExecutionError(response.message)
            if not isinstance(response, ScheduleDecision):
                self._enabled = False
                raise SchedulerExecutionError(
                    "scheduler must return a ScheduleDecision"
                )
            return SchedulerResult(
                response,
                asyncio.get_running_loop().time() - started_s,
            )

    def close(self) -> None:
        self._enabled = False
        self._requests.put(_STOP)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is _STOP:
                return
            request_id, fleet, costs = request
            try:
                response: object = self._scheduler.schedule(fleet, costs)
            except BaseException as error:
                response = _SchedulerFailure(f"{type(error).__name__}: {error}")
            self._responses.put((request_id, response))


@dataclass(frozen=True, slots=True)
class _SchedulerFailure:
    message: str


_STOP = object()
