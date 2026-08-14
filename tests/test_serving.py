import asyncio
import threading
import time

import pytest

from fleetvla import (
    ActionChunk,
    FieldSpec,
    FIFOScheduler,
    RemoteActionReceipt,
    ScheduleDecision,
    SessionConfig,
    SyntheticBackend,
)
from fleetvla.backend import BackendResult
from fleetvla.serving import AsyncServingEngine, serving_metrics


class FakeEndpoint:
    observation_schema = (FieldSpec("state", (1,)),)

    def __init__(self, session_id: str, *, fail_action: bool = False) -> None:
        self.session_config = SessionConfig(
            session_id, control_hz=50, chunk_size=2, request_threshold_s=0.02
        )
        self.observations = 0
        self.actions = []
        self.fallbacks = 0
        self.closed = False
        self.fail_action = fail_action

    def observe(self):
        self.observations += 1
        return {"state": ShapeValue((1,), self.observations)}

    def execute(self, action):
        if self.fail_action:
            raise RuntimeError("motor fault")
        self.actions.append(action)

    def fallback(self):
        self.fallbacks += 1

    def close(self):
        self.closed = True

    def reconnect(self):
        self.closed = False


class ShapeValue:
    def __init__(self, shape, value):
        self.shape = shape
        self.value = value


class SleepingBackend(SyntheticBackend):
    def infer(self, observations, started_at_s):
        time.sleep(0.04)
        return super().infer(observations, started_at_s)


class SlowEndpoint(FakeEndpoint):
    def execute(self, action):
        time.sleep(0.04)
        super().execute(action)

    def fallback(self):
        time.sleep(0.04)
        super().fallback()


class ShutdownProbeEndpoint(FakeEndpoint):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.operation_active = False
        self.close_overlapped = False

    def execute(self, action):
        self.operation_active = True
        try:
            time.sleep(0.06)
            super().execute(action)
        finally:
            self.operation_active = False

    def close(self):
        self.close_overlapped = self.operation_active
        super().close()


class BlockingScheduler(FIFOScheduler):
    def schedule(self, fleet, costs):
        time.sleep(0.2)
        return super().schedule(fleet, costs)


class CommandEndpoint(FakeEndpoint):
    def __init__(self, session_id: str, *, acknowledgement_delay_s: float) -> None:
        super().__init__(session_id)
        self.acknowledgement_delay_s = acknowledgement_delay_s

    async def execute_command(self, command):
        await asyncio.sleep(self.acknowledgement_delay_s)
        self.actions.append(command.value)
        return RemoteActionReceipt(accepted=True, executed=True)


class AsyncShutdownProbeEndpoint(CommandEndpoint):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id, acknowledgement_delay_s=0.05)
        self.operation_active = False
        self.close_overlapped = False

    async def execute_command(self, command):
        self.operation_active = True
        try:
            return await super().execute_command(command)
        finally:
            self.operation_active = False

    def close(self):
        self.close_overlapped = self.operation_active
        super().close()


class BrieflyDeferredScheduler(FIFOScheduler):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def schedule(self, fleet, costs):
        self.calls += 1
        oldest_request_s = min(
            session.request_time_s for session in fleet.ready_sessions
        )
        dispatch_at_s = oldest_request_s + 0.02
        if fleet.now_s < dispatch_at_s:
            return ScheduleDecision((), "wait briefly", defer_until_s=dispatch_at_s)
        return super().schedule(fleet, costs)


class InvalidScheduler:
    def schedule(self, fleet, costs):
        del fleet, costs
        return ScheduleDecision(("alien",), "invalid session")


class FailingBackend(SyntheticBackend):
    def infer(self, observations, started_at_s):
        raise RuntimeError("GPU unavailable")


class EmptyBackend(SyntheticBackend):
    def infer(self, observations, started_at_s):
        return BackendResult(0, ())


class UnknownSessionBackend(SyntheticBackend):
    def infer(self, observations, started_at_s):
        observation = tuple(observations)[0]
        return BackendResult(
            0,
            (
                ActionChunk(
                    "unknown",
                    observation.sequence,
                    observation.generation,
                    (1, 2),
                    started_at_s,
                ),
            ),
        )


class ConcurrentProbeBackend(SyntheticBackend):
    def __init__(self):
        super().__init__(chunk_size=2, base_latency_s=0, per_item_latency_s=0)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def infer(self, observations, started_at_s):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.04)
            return super().infer(observations, started_at_s)
        finally:
            with self.lock:
                self.active -= 1


def test_wall_clock_control_ticks_continue_during_batched_inference() -> None:
    endpoints = [FakeEndpoint("a"), FakeEndpoint("b")]
    engine = AsyncServingEngine(
        endpoints,
        SleepingBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
        max_batch_size=2,
    )

    events = asyncio.run(engine.run(0.14))

    starts = [event for event in events if event.kind == "batch_dispatched"]
    assert starts[0].details["batch_size"] == 2
    completed_at = next(
        event.time_s for event in events if event.kind == "inference_completed"
    )
    assert any(
        event.kind == "endpoint_fallback" and event.time_s < completed_at
        for event in events
    )
    assert all(endpoint.actions for endpoint in endpoints)
    metrics = serving_metrics(events, endpoints, 0.14)
    assert metrics.backend_utilization > 0
    cost_event = next(
        event for event in events if event.kind == "scheduler_cost_estimate"
    )
    assert cost_event.details["estimated_latency_s"] == 0
    completed = next(event for event in events if event.kind == "inference_completed")
    assert "backend_reported_latency_s" in completed.details


def test_wall_clock_scheduler_can_defer_without_busy_looping() -> None:
    scheduler = BrieflyDeferredScheduler()
    engine = AsyncServingEngine(
        [FakeEndpoint("a"), FakeEndpoint("b")],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        scheduler,
        max_batch_size=2,
    )

    events = asyncio.run(engine.run(0.06))

    deferred = next(event for event in events if event.kind == "dispatch_deferred")
    dispatched = next(event for event in events if event.kind == "batch_dispatched")
    assert dispatched.time_s >= deferred.details["defer_until_s"]
    assert dispatched.details["batch_size"] == 2
    assert scheduler.calls < 10


def test_wall_clock_metrics_count_control_ticks_missed_by_slow_endpoint() -> None:
    endpoint = SlowEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
    )

    events = asyncio.run(engine.run(0.12))

    missed = sum(
        event.details["count"]
        for event in events
        if event.kind == "control_ticks_missed"
    )
    ordinary_starvation = sum(event.kind == "action_starved" for event in events)
    actions = sum(event.kind == "action_executed" for event in events)
    metrics = serving_metrics(events, [endpoint], 0.12)
    assert missed > 0
    assert metrics.per_session["a"]["starved_ticks"] == (missed + ordinary_starvation)
    assert metrics.starvation_frequency == (missed + ordinary_starvation) / (
        missed + ordinary_starvation + actions
    )


def test_wall_clock_does_not_count_ticks_beyond_run_end() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        BlockingScheduler(),
    )

    events = asyncio.run(engine.run(0.1))

    accounted = sum(
        event.kind in {"action_executed", "action_starved"} for event in events
    ) + sum(
        event.details["count"]
        for event in events
        if event.kind == "control_ticks_missed"
    )
    assert accounted == 5
    assert any(event.kind == "scheduler_failed" for event in events)
    decision = next(event for event in events if event.kind == "scheduler_decision")
    assert decision.details["fallback"] is True


def test_scheduler_timeout_does_not_block_control_fallbacks() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        BlockingScheduler(),
        scheduler_timeout_s=0.005,
    )

    started = time.monotonic()
    events = asyncio.run(engine.run(0.08))

    assert time.monotonic() - started < 0.18
    failure = next(event for event in events if event.kind == "scheduler_failed")
    assert "decision budget" in failure.details["error"]
    assert any(event.kind == "action_executed" for event in events)
    assert any(event.kind == "batch_dispatched" for event in events)


def test_invalid_scheduler_decision_falls_back_without_poisoning_trace() -> None:
    engine = AsyncServingEngine(
        [FakeEndpoint("a")],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        InvalidScheduler(),
    )

    events = asyncio.run(engine.run(0.06))

    assert any(event.kind == "scheduler_failed" for event in events)
    decisions = [event for event in events if event.kind == "scheduler_decision"]
    assert decisions
    assert all(event.details["selected_session_ids"] == ("a",) for event in decisions)
    assert all(event.details["fallback"] is True for event in decisions)


@pytest.mark.parametrize("timeout_s", [0, float("nan"), float("inf")])
def test_scheduler_timeout_must_be_finite_and_positive(timeout_s) -> None:
    with pytest.raises(ValueError, match="scheduler_timeout_s"):
        AsyncServingEngine(
            [FakeEndpoint("a")],
            SyntheticBackend(),
            FIFOScheduler(),
            scheduler_timeout_s=timeout_s,
        )


def test_endpoint_action_failure_disconnects_and_rejects_command() -> None:
    endpoint = FakeEndpoint("a", fail_action=True)
    engine = AsyncServingEngine(
        [endpoint],
        SleepingBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
    )

    events = asyncio.run(engine.run(0.1))

    assert any(event.kind == "action_rejected_endpoint" for event in events)
    assert any(event.kind == "session_disconnected" for event in events)
    assert endpoint.closed


def test_close_waits_for_timed_out_endpoint_worker() -> None:
    endpoint = ShutdownProbeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
        endpoint_timeout_s=0.005,
    )

    asyncio.run(engine.run(0.04))
    engine.close()

    assert endpoint.closed
    assert not endpoint.close_overlapped


def test_async_close_drains_timed_out_coroutine_endpoint() -> None:
    endpoint = AsyncShutdownProbeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
        endpoint_timeout_s=0.005,
    )

    async def run_and_close() -> None:
        await engine.run(0.04)
        await engine.aclose()

    asyncio.run(run_and_close())

    assert endpoint.closed
    assert not endpoint.close_overlapped


def test_reconnect_waits_for_a_fresh_observation() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
    )

    asyncio.run(engine._observe("a"))
    original = engine.runtime.snapshot().sessions[0]
    asyncio.run(engine._disconnect("a"))
    assert endpoint.closed
    assert not engine.runtime.snapshot().sessions[0].connected

    asyncio.run(engine.reconnect_session("a"))

    session = engine.runtime.snapshot().sessions[0]
    assert session.connected
    assert session.ready_sequence == 1
    assert session.generation == original.generation + 1
    assert endpoint.observations == 2
    assert not endpoint.closed


def test_remote_ack_timeout_does_not_block_healthy_session() -> None:
    blocked = CommandEndpoint("blocked", acknowledgement_delay_s=0.2)
    healthy = CommandEndpoint("healthy", acknowledgement_delay_s=0)
    engine = AsyncServingEngine(
        [blocked, healthy],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
        max_batch_size=2,
        endpoint_timeout_s=0.02,
    )

    events = asyncio.run(engine.run(0.12))

    assert not any(
        event.kind == "action_executed" and event.session_id == "blocked"
        for event in events
    )
    assert any(
        event.kind == "action_rejected_endpoint" and event.session_id == "blocked"
        for event in events
    )
    assert any(
        event.kind == "action_executed" and event.session_id == "healthy"
        for event in events
    )
    metrics = serving_metrics(events, [blocked, healthy], 0.12)
    assert metrics.sent_actions > metrics.accepted_actions
    assert metrics.accepted_actions == metrics.useful_actions


def test_reset_rejects_remote_ack_from_previous_generation() -> None:
    endpoint = CommandEndpoint("arm", acknowledgement_delay_s=0.04)
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
    )

    async def run_and_reset():
        run = asyncio.create_task(engine.run(0.07))
        await asyncio.sleep(0.03)
        engine.reset_session("arm")
        return await run

    events = asyncio.run(run_and_reset())

    assert any(event.kind == "action_ack_rejected_stale" for event in events)
    assert not any(event.kind == "action_executed" for event in events)


def test_backend_failure_resets_generation_and_uses_local_fallback() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint], FailingBackend(chunk_size=2), FIFOScheduler()
    )

    events = asyncio.run(engine.run(0.06))

    assert any(event.kind == "inference_failed" for event in events)
    assert any(event.kind == "session_reset" for event in events)
    assert endpoint.fallbacks > 0


@pytest.mark.parametrize("backend_type", [EmptyBackend, UnknownSessionBackend])
def test_malformed_backend_result_uses_batch_failure_path(backend_type) -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine([endpoint], backend_type(chunk_size=2), FIFOScheduler())

    events = asyncio.run(engine.run(0.06))

    assert any(event.kind == "inference_failed" for event in events)
    assert any(event.kind == "session_reset" for event in events)
    assert endpoint.fallbacks > 0
    snapshot = engine.runtime.snapshot().sessions[0]
    assert snapshot.in_flight_sequence is None


def test_inference_timeout_resets_batch_and_falls_back() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SleepingBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
        inference_timeout_s=0.005,
    )

    events = asyncio.run(engine.run(0.04))

    assert any(event.kind == "inference_failed" for event in events)
    assert endpoint.fallbacks > 0


def test_timed_out_worker_is_quarantined_until_it_exits() -> None:
    endpoint = FakeEndpoint("a")
    backend = ConcurrentProbeBackend()
    engine = AsyncServingEngine(
        [endpoint],
        backend,
        FIFOScheduler(),
        inference_timeout_s=0.005,
    )

    events = asyncio.run(engine.run(0.12))

    assert sum(event.kind == "inference_failed" for event in events) >= 2
    assert backend.calls >= 2
    assert backend.max_active == 1


@pytest.mark.parametrize("duration_s", [0, float("nan"), float("inf"), -float("inf")])
def test_serving_metrics_duration_must_be_finite_and_positive(duration_s) -> None:
    with pytest.raises(ValueError, match="metrics duration"):
        serving_metrics((), [FakeEndpoint("a")], duration_s)
