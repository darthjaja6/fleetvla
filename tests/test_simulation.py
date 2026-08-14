import math

import pytest

from fleetvla import (
    FleetSimulator,
    RobotSpec,
    ScheduleDecision,
    SyntheticBackend,
)


class CoalescingScheduler:
    def schedule(self, fleet, costs):
        del costs
        ready = fleet.ready_sessions
        if not ready:
            return ScheduleDecision((), "no work")
        oldest_request_s = min(session.request_time_s for session in ready)
        dispatch_at_s = oldest_request_s + 0.02
        if fleet.now_s < dispatch_at_s:
            return ScheduleDecision((), "wait for peers", defer_until_s=dispatch_at_s)
        return ScheduleDecision(
            tuple(session.session_id for session in ready), "coalesced"
        )


def test_scheduler_can_defer_to_coalesce_a_later_request() -> None:
    result = FleetSimulator(
        [
            RobotSpec("first", control_hz=10, chunk_size=1),
            RobotSpec("second", control_hz=10, chunk_size=1),
        ],
        scheduler=CoalescingScheduler(),
        backend=SyntheticBackend(chunk_size=1, base_latency_s=0, per_item_latency_s=0),
        max_batch_size=2,
        observation_schedule=((0.0, "first"), (0.01, "second")),
    ).run(0.1)

    dispatch = next(
        event for event in result.events if event.kind == "batch_dispatched"
    )
    assert dispatch.time_s == 0.02
    assert dispatch.details["session_ids"] == ("first", "second")
    assert result.count("dispatch_deferred") == 2


def test_simulation_is_deterministic_and_batches_simultaneous_requests() -> None:
    robots = [
        RobotSpec("fast", control_hz=20, chunk_size=4),
        RobotSpec("slow", control_hz=10, chunk_size=4),
    ]

    first = FleetSimulator(robots, max_batch_size=2).run(1.0)
    second = FleetSimulator(robots, max_batch_size=2).run(1.0)

    assert first.events == second.events
    dispatches = [event for event in first.events if event.kind == "request_dispatched"]
    assert dispatches[0].details["batch_size"] == 2
    assert first.count("action_executed", "fast") > 0
    assert first.count("action_executed", "slow") > 0


def test_slow_backend_causes_starvation_without_blocking_virtual_time() -> None:
    result = FleetSimulator(
        [RobotSpec("arm", control_hz=20, chunk_size=2)],
        backend=SyntheticBackend(
            chunk_size=2, base_latency_s=0.2, per_item_latency_s=0.0
        ),
    ).run(0.5)

    assert result.count("action_starved") > 0
    assert result.count("chunk_accepted") > 0


def test_integer_tick_ordinals_preserve_coincidences_and_end_boundary() -> None:
    robots = [
        RobotSpec("fast", control_hz=20, chunk_size=1, request_threshold_s=1),
        RobotSpec("slow", control_hz=10, chunk_size=1, request_threshold_s=1),
    ]
    result = FleetSimulator(
        robots,
        backend=SyntheticBackend(chunk_size=1, base_latency_s=0, per_item_latency_s=0),
        max_batch_size=2,
    ).run(0.5)

    slow_ticks = [
        event.time_s
        for event in result.events
        if event.kind in {"action_executed", "action_starved"}
        and event.session_id == "slow"
    ]
    assert slow_ticks == [0.1, 0.2, 0.3, 0.4, 0.5]
    coincident_batches = [
        event.details["batch_size"]
        for event in result.events
        if event.kind == "request_dispatched"
        and event.session_id == "fast"
        and event.time_s in {0.1, 0.2, 0.3, 0.4, 0.5}
    ]
    assert coincident_batches == [2, 2, 2, 2, 2]


def test_zero_delay_delivery_precedes_control_tick_at_same_time() -> None:
    result = FleetSimulator(
        [RobotSpec("arm", control_hz=5, chunk_size=2)],
        backend=SyntheticBackend(
            chunk_size=2, base_latency_s=0.2, per_item_latency_s=0
        ),
    ).run(0.2)

    at_boundary = [event.kind for event in result.events if event.time_s == 0.2]
    assert at_boundary.index("chunk_accepted") < at_boundary.index("action_executed")
    assert "action_starved" not in at_boundary


def test_backend_horizon_mismatch_is_rejected() -> None:
    result = FleetSimulator(
        [RobotSpec("arm", control_hz=10, chunk_size=2)],
        backend=SyntheticBackend(chunk_size=5, base_latency_s=0, per_item_latency_s=0),
    ).run(0.1)

    assert result.count("chunk_rejected_horizon") > 0
    assert result.count("action_executed") == 0


def test_simulator_is_explicitly_single_use() -> None:
    simulator = FleetSimulator([RobotSpec("arm", control_hz=10)])
    simulator.run(0.1)

    try:
        simulator.run(0.2)
    except RuntimeError as error:
        assert "only be run once" in str(error)
    else:
        raise AssertionError("simulator silently reused stale event state")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_simulation_timing_must_be_finite(value) -> None:
    with pytest.raises(ValueError):
        RobotSpec("arm", control_hz=10, start_delay_s=value)
    simulator = FleetSimulator([RobotSpec("arm", control_hz=10)])
    with pytest.raises(ValueError):
        simulator.run(value)
    with pytest.raises(ValueError, match="observation times"):
        FleetSimulator(
            [RobotSpec("arm", control_hz=10)],
            observation_schedule=((value, "arm"),),
        )
