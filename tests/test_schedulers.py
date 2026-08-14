from pathlib import Path

import pytest

from fleetvla import (
    FleetSnapshot,
    InferenceCostModel,
    ScheduleDecision,
    SessionSnapshot,
)
from fleetvla.cli import main
from fleetvla.schedulers import check_scheduler, create_scheduler, registry


def _session(
    session_id: str,
    *,
    request_time_s: float,
    buffer_horizon_s: float,
    network_latency_s: float = 0,
    chunk_size: int = 1,
    service_weight: float = 1,
) -> SessionSnapshot:
    return SessionSnapshot(
        session_id=session_id,
        generation=0,
        ready_sequence=0,
        request_time_s=request_time_s,
        buffer_steps=int(buffer_horizon_s * 20),
        buffer_horizon_s=buffer_horizon_s,
        control_hz=20,
        latency_budget_s=0.2,
        network_latency_s=network_latency_s,
        connected=True,
        in_flight_sequence=None,
        chunk_size=chunk_size,
        service_weight=service_weight,
    )


def test_all_registered_schedulers_pass_public_conformance() -> None:
    assert registry.names() == (
        "adaptive-slack",
        "edf",
        "fifo",
        "lookahead",
        "round-robin",
    )
    for name in registry.names():
        assert len(check_scheduler(lambda name=name: create_scheduler(name))) == 7


def test_conformance_accepts_future_dispatch_deferral() -> None:
    class Deferred:
        def schedule(self, fleet, costs):
            del costs
            if not fleet.ready_sessions:
                return ScheduleDecision((), "no work")
            return ScheduleDecision((), "wait for peers", fleet.now_s + 0.01)

    assert len(check_scheduler(Deferred)) == 7


def test_conformance_rejects_non_future_dispatch_deferral() -> None:
    class InvalidDeferred:
        def schedule(self, fleet, costs):
            del costs
            return ScheduleDecision((), "not future", fleet.now_s)

    with pytest.raises(ValueError, match="later than"):
        check_scheduler(InvalidDeferred)


def test_builtin_algorithms_make_distinct_documented_choices() -> None:
    fleet = FleetSnapshot(
        now_s=1.0,
        sessions=(
            _session("old-safe", request_time_s=0.85, buffer_horizon_s=0.4),
            _session("new-urgent", request_time_s=0.9, buffer_horizon_s=0.01),
        ),
        max_batch_size=2,
    )
    costs = InferenceCostModel(0.02, 0.01)

    fifo = create_scheduler("fifo", {"batch_size_limit": 1})
    edf = create_scheduler("edf", {"batch_size_limit": 1})
    adaptive = create_scheduler("adaptive-slack")

    assert fifo.schedule(fleet, costs).session_ids == ("old-safe",)
    assert edf.schedule(fleet, costs).session_ids == ("new-urgent",)
    assert adaptive.schedule(fleet, costs).session_ids == ("new-urgent",)


def test_round_robin_rotates_when_only_one_session_can_be_selected() -> None:
    fleet = FleetSnapshot(
        0,
        (
            _session("a", request_time_s=0, buffer_horizon_s=0),
            _session("b", request_time_s=0, buffer_horizon_s=0),
        ),
        2,
    )
    scheduler = create_scheduler("round-robin", {"batch_size_limit": 1})
    costs = InferenceCostModel(0, 0)

    assert scheduler.schedule(fleet, costs).session_ids == ("a",)
    assert scheduler.schedule(fleet, costs).session_ids == ("b",)


def test_lookahead_models_weighted_execution_and_dynamic_batch_size() -> None:
    fleet = FleetSnapshot(
        1,
        (
            _session(
                "fast",
                request_time_s=1,
                buffer_horizon_s=0,
                chunk_size=6,
                service_weight=5,
            ),
            _session(
                "slow",
                request_time_s=1,
                buffer_horizon_s=0.5,
                chunk_size=10,
            ),
        ),
        2,
    )
    costs = InferenceCostModel(0, 0, (0.05, 0.2))

    decision = create_scheduler("lookahead").schedule(fleet, costs)

    assert decision.session_ids == ("fast",)
    assert "weighted executed time" in decision.reason


def test_measured_latency_profile_must_cover_requested_batch() -> None:
    costs = InferenceCostModel(0.02, 0.01, (0.05,))

    assert costs.estimate(1) == 0.05
    with pytest.raises(ValueError, match="exceeds"):
        costs.estimate(2)


def test_local_scheduler_loads_without_registry_edit() -> None:
    path = Path(__file__).parents[1] / "examples" / "my_scheduler.py"
    scheduler = create_scheduler(f"{path}:SmallestBufferFirst")
    assert len(check_scheduler(lambda: scheduler)) == 7


def test_unknown_typed_config_fails_before_a_run() -> None:
    with pytest.raises(ValueError, match="unknown config"):
        create_scheduler("fifo", {"transport_margin_s": 1})


@pytest.mark.parametrize(
    ("scheduler", "config", "message"),
    [
        ("fifo", {"batch_size_limit": "two"}, "must be an integer"),
        ("fifo", {"batch_size_limit": True}, "must be an integer"),
        (
            "adaptive-slack",
            {"transport_margin_s": "slow"},
            "must be a number",
        ),
    ],
)
def test_typed_config_rejects_wrong_json_types(
    scheduler: str, config: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_scheduler(scheduler, config)


def test_conformance_rejects_structural_imitation() -> None:
    class InvalidScheduler:
        def schedule(self, fleet, costs):
            del fleet, costs
            return type(
                "Decision", (), {"session_ids": ("arm-a", "arm-a"), "reason": ""}
            )()

    with pytest.raises(ValueError, match="ScheduleDecision"):
        check_scheduler(InvalidScheduler)


def test_conformance_cli_reports_contract_error_without_traceback(
    tmp_path, capsys
) -> None:
    path = tmp_path / "invalid.py"
    path.write_text(
        "class Invalid:\n"
        "    def schedule(self, fleet, costs):\n"
        "        return ('not', 'a', 'decision')\n"
    )

    assert main(["test-scheduler", f"{path}:Invalid"]) == 2
    error = capsys.readouterr().err
    assert error == "error: scheduler must return a ScheduleDecision\n"


def test_conformance_rejects_hard_coded_fixture_session() -> None:
    class HardCoded:
        def schedule(self, fleet, costs):
            del fleet, costs
            from fleetvla import ScheduleDecision

            return ScheduleDecision(("arm-a",), "hard coded")

    with pytest.raises(ValueError, match="not ready"):
        check_scheduler(HardCoded)
