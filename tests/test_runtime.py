import math

import pytest

from fleetvla import (
    ActionCommand,
    ActionChunk,
    FleetRuntime,
    ScheduleDecision,
    SessionConfig,
    VirtualClock,
)


def test_scheduler_only_sees_immutable_snapshot_and_chunk_is_consumed() -> None:
    clock = VirtualClock()
    runtime = FleetRuntime(clock, max_batch_size=2)
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=2))
    observation = runtime.observe("arm", {"joint": 1})

    snapshot = runtime.snapshot()
    assert snapshot.ready_sessions[0].session_id == "arm"
    batch = runtime.prepare_batch(ScheduleDecision(("arm",)))
    assert batch == (observation,)

    clock.advance(0.03)
    accepted = runtime.accept(
        ActionChunk("arm", observation.sequence, 0, ("left", "right"), clock.now())
    )
    assert accepted
    assert runtime.consume_action("arm") == "left"
    assert runtime.snapshot().sessions[0].buffer_steps == 1


def test_reset_rejects_result_from_previous_generation() -> None:
    clock = VirtualClock()
    runtime = FleetRuntime(clock)
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=2))
    observation = runtime.observe("arm")
    runtime.prepare_batch(ScheduleDecision(("arm",)))
    runtime.reset("arm")

    stale = ActionChunk("arm", observation.sequence, 0, (1, 2), clock.now())
    assert not runtime.accept(stale)
    assert runtime.snapshot().sessions[0].buffer_steps == 0
    assert runtime.events.events[-1].kind == "chunk_rejected_stale"


def test_chunk_must_match_declared_action_horizon() -> None:
    clock = VirtualClock()
    runtime = FleetRuntime(clock)
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=2))
    observation = runtime.observe("arm")
    runtime.prepare_batch(ScheduleDecision(("arm",)))

    oversized = ActionChunk("arm", observation.sequence, 0, (1, 2, 3), clock.now())
    assert not runtime.accept(oversized)
    assert runtime.snapshot().sessions[0].buffer_steps == 0
    assert runtime.events.events[-1].kind == "chunk_rejected_horizon"
    assert runtime.events.events[-1].details == {
        "expected_actions": 2,
        "received_actions": 3,
    }


def test_disconnect_requires_explicit_reconnect() -> None:
    runtime = FleetRuntime(VirtualClock())
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=2))
    runtime.disconnect("arm")

    try:
        runtime.observe("arm")
    except RuntimeError as error:
        assert "disconnected" in str(error)
    else:
        raise AssertionError("disconnected session accepted an observation")

    runtime.reconnect("arm")
    assert runtime.observe("arm").generation == 1


def test_disconnect_rejects_chunk_from_previous_generation() -> None:
    clock = VirtualClock()
    runtime = FleetRuntime(clock)
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=2))
    observation = runtime.observe("arm")
    runtime.prepare_batch(ScheduleDecision(("arm",)))
    runtime.disconnect("arm")
    runtime.reconnect("arm")

    stale = ActionChunk("arm", observation.sequence, 0, (1, 2), clock.now())
    assert not runtime.accept(stale)
    assert runtime.snapshot().sessions[0].buffer_steps == 0


def test_action_acknowledgements_reject_duplicates_and_stale_generations() -> None:
    clock = VirtualClock()
    runtime = FleetRuntime(clock)
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=1))
    observation = runtime.observe("arm")
    runtime.prepare_batch(ScheduleDecision(("arm",)))
    runtime.accept(ActionChunk("arm", 0, 0, (1,), clock.now()))
    command = runtime.dequeue_action("arm")
    assert command is not None

    assert runtime.acknowledge(command, accepted=True)
    assert not runtime.acknowledge(command, accepted=True)
    assert runtime.events.events[-1].kind == "action_ack_rejected_duplicate"

    runtime.reset("arm")
    assert not runtime.acknowledge(command, accepted=True)
    assert runtime.events.events[-1].kind == "action_ack_rejected_stale"


def test_fabricated_acknowledgement_is_rejected() -> None:
    runtime = FleetRuntime(VirtualClock())
    runtime.register(SessionConfig("arm", control_hz=10, chunk_size=1))
    fabricated = ActionCommand("arm", 0, 999, 42, 0, 0)

    assert not runtime.acknowledge(fabricated, accepted=True)
    assert runtime.events.events[-1].kind == "action_ack_rejected_unexpected"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_session_and_cost_timing_must_be_finite(value) -> None:
    with pytest.raises(ValueError):
        SessionConfig("arm", control_hz=value, chunk_size=1)
    with pytest.raises(ValueError):
        SessionConfig(
            "arm", control_hz=10, chunk_size=1, network_latency_s=value
        )
    from fleetvla import InferenceCostModel

    with pytest.raises(ValueError):
        InferenceCostModel(value, 0)
    with pytest.raises(ValueError):
        VirtualClock(value)
