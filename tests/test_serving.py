import asyncio
import time
import threading

import pytest

from fleetvla import (
    ActionChunk,
    FIFOScheduler,
    FieldSpec,
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


class BlockingScheduler(FIFOScheduler):
    def schedule(self, fleet, costs):
        time.sleep(0.2)
        return super().schedule(fleet, costs)


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
    assert metrics.per_session["a"]["starved_ticks"] == (
        missed + ordinary_starvation
    )
    assert metrics.starvation_frequency == (
        missed + ordinary_starvation
    ) / (missed + ordinary_starvation + actions)


def test_wall_clock_does_not_count_ticks_beyond_run_end() -> None:
    endpoint = FakeEndpoint("a")
    engine = AsyncServingEngine(
        [endpoint],
        SyntheticBackend(chunk_size=2, base_latency_s=0, per_item_latency_s=0),
        BlockingScheduler(),
    )

    events = asyncio.run(engine.run(0.1))

    accounted = sum(
        event.kind in {"action_executed", "action_starved"}
        for event in events
    ) + sum(
        event.details["count"]
        for event in events
        if event.kind == "control_ticks_missed"
    )
    assert accounted == 5


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
    engine = AsyncServingEngine(
        [endpoint], backend_type(chunk_size=2), FIFOScheduler()
    )

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
