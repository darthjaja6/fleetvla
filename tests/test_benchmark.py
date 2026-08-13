import hashlib
import json
import shutil
import math
from dataclasses import replace
from pathlib import Path

import pytest

from fleetvla.benchmark import (
    artifact_dict,
    default_config,
    fleetvla_source_sha256,
    load_artifact,
    load_config,
    replay_artifact,
    run_benchmark,
    trace_config,
    verify_artifact,
    write_artifact,
)
from fleetvla.cli import main


def _seal_artifact(body):
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _event(artifact, kind):
    return next(event for event in artifact["events"] if event["kind"] == kind)


def _remove_event(artifact, kind):
    artifact["events"].remove(_event(artifact, kind))


def _replace_chunk_with_equal_horizon_rejection(artifact):
    event = _event(artifact, "chunk_accepted")
    event["kind"] = "chunk_rejected_horizon"
    event["details"] = {"expected_actions": 8, "received_actions": 8}


def _forge_action_age(artifact):
    _event(artifact, "action_executed")["details"]["action_age_s"] = 0.9
    artifact["metrics"]["system"].update(
        action_age_p50_s=0.9, action_age_p95_s=0.9
    )


def _forge_inference_latency(artifact):
    _event(artifact, "inference_completed")["details"]["latency_s"] = 0.9
    artifact["metrics"]["system"]["backend_utilization"] = 0.9


def _forge_backend_latency(artifact):
    _event(artifact, "inference_completed")["details"][
        "backend_reported_latency_s"
    ] = 0.9


def _forge_chunk_age(artifact):
    _event(artifact, "chunk_accepted")["details"]["action_age_s"] = 0.9


def _replace_chunk_with_wrong_expected_horizon(artifact):
    event = _event(artifact, "chunk_accepted")
    event["kind"] = "chunk_rejected_horizon"
    event["details"] = {"expected_actions": 9, "received_actions": 10}


def _valid_system_artifact():
    session_id = "libero_spatial-0"
    return _seal_artifact(
        {
            "artifact_version": 1,
            "artifact_kind": "system",
            "provenance": {
                "fleetvla_version": "0.1.0",
                "fleetvla_source_sha256": fleetvla_source_sha256(),
                "lerobot_version": "0.4.0",
                "libero_version": "0.1.4",
                "python_version": "3.11.0",
                "platform": "test-platform",
                "torch_version": "2.0.0",
                "cuda_version": None,
                "accelerator": "cpu",
                "scheduler_source": None,
                "scheduler_source_sha256": None,
            },
            "config": {
                "clock": "monotonic-wall",
                "environment": "libero",
                "suite": "libero_spatial",
                "task_ids": [0],
                "task_descriptions": {session_id: "pick up the object"},
                "model_id": "example/model",
                "model_revision": "b" * 40,
                "scheduler": "edf",
                "scheduler_config": {},
                "duration_s": 1.0,
                "max_batch_size": 1,
                "control_hz": 1.0,
                "model_output_horizon": 50,
                "execution_horizon": 8,
                "action_execution": "sequential-buffer",
                "episode_length": 280,
                "seed": 0,
            },
            "metrics": {
                "system": {
                    "useful_actions": 1,
                    "starvation_frequency": 0.0,
                    "starvation_duration_s": 0.0,
                    "action_age_p50_s": 0.1,
                    "action_age_p95_s": 0.1,
                    "fairness": 1.0,
                    "batch_sizes": [1],
                    "backend_utilization": 0.04,
                    "per_session": {
                        session_id: {
                            "actions": 1,
                            "starved_ticks": 0,
                            "starvation_duration_s": 0.0,
                            "useful_progress_ratio": 1.0,
                        }
                    },
                },
                "tasks": {
                    session_id: {
                        "reward": 1.0,
                        "steps": 1,
                        "successes": 1,
                        "episodes": 1,
                        "terminated": 1,
                        "truncated": 0,
                    }
                },
            },
            "events": [
                {
                    "time_s": 0.0,
                    "kind": "session_registered",
                    "session_id": session_id,
                    "details": None,
                },
                {
                    "time_s": 0.001,
                    "kind": "observation_ready",
                    "session_id": session_id,
                    "details": {"sequence": 0},
                },
                {
                    "time_s": 0.005,
                    "kind": "request_dispatched",
                    "session_id": session_id,
                    "details": {"sequence": 0, "batch_size": 1},
                },
                {
                    "time_s": 0.01,
                    "kind": "batch_dispatched",
                    "session_id": None,
                    "details": {
                        "batch_size": 1,
                        "session_ids": [session_id],
                        "reason": "test",
                        "selected_state": [
                            {
                                "session_id": session_id,
                                "buffer_horizon_s": 0.0,
                                "request_age_s": 0.01,
                            }
                        ],
                    },
                },
                {
                    "time_s": 0.011,
                    "kind": "scheduler_cost_estimate",
                    "session_id": None,
                    "details": {
                        "batch_size": 1,
                        "base_latency_s": 0.03,
                        "per_item_latency_s": 0.01,
                        "estimated_latency_s": 0.04,
                    },
                },
                {
                    "time_s": 0.012,
                    "kind": "inference_started",
                    "session_id": None,
                    "details": {"batch_size": 1},
                },
                {
                    "time_s": 0.052,
                    "kind": "inference_completed",
                    "session_id": None,
                    "details": {
                        "batch_size": 1,
                        "latency_s": 0.04,
                        "backend_reported_latency_s": 0.04,
                        "output_shapes": [[1, 50, 7]],
                        "execution_horizons": [8],
                    },
                },
                {
                    "time_s": 0.06,
                    "kind": "chunk_accepted",
                    "session_id": session_id,
                    "details": {
                        "sequence": 0,
                        "actions": 8,
                        "action_age_s": 0.008,
                    },
                },
                {
                    "time_s": 0.1,
                    "kind": "action_dequeued",
                    "session_id": session_id,
                    "details": {
                        "remaining_steps": 7,
                        "sequence": 0,
                        "action_index": 0,
                        "action_age_s": 0.1,
                    },
                },
                {
                    "time_s": 0.1,
                    "kind": "action_executed",
                    "session_id": session_id,
                    "details": {
                        "sequence": 0,
                        "action_index": 0,
                        "action_age_s": 0.1,
                    },
                },
                {
                    "time_s": 0.2,
                    "kind": "endpoint_task_step",
                    "session_id": session_id,
                    "details": {
                        "reward": 1.0,
                        "success": True,
                        "terminated": True,
                        "truncated": False,
                    },
                },
                {
                    "time_s": 0.2,
                    "kind": "endpoint_episode_boundary",
                    "session_id": session_id,
                    "details": None,
                },
                {
                    "time_s": 0.2,
                    "kind": "session_reset",
                    "session_id": session_id,
                    "details": {"generation": 1},
                },
            ],
        }
    )


def test_benchmark_produces_multidimensional_metrics() -> None:
    run = run_benchmark(default_config())

    assert run.metrics.useful_actions > 0
    assert 0 <= run.metrics.starvation_frequency <= 1
    assert run.metrics.action_age_p95_s is not None
    assert 0 <= run.metrics.fairness <= 1
    assert run.metrics.batch_sizes
    assert 0 <= run.metrics.backend_utilization <= 1
    assert set(run.metrics.per_session) == {"fast-arm", "slow-arm"}


def test_artifact_replays_every_event_and_metric(tmp_path) -> None:
    run = run_benchmark(default_config())
    path = write_artifact(run, tmp_path / "run.json")

    replayed, matches = replay_artifact(path)

    assert matches
    assert replayed.metrics == run.metrics
    assert load_artifact(path)["sha256"] == artifact_dict(run)["sha256"]
    verified = verify_artifact(path)
    assert verified.get("artifact_kind", "algorithm") == "algorithm"
    assert len(verified["provenance"]["fleetvla_source_sha256"]) == 64


def test_system_artifact_verifies_but_does_not_replay(tmp_path) -> None:
    body = _valid_system_artifact()
    path = tmp_path / "system.json"
    path.write_text(json.dumps(body))

    assert verify_artifact(path)["artifact_kind"] == "system"
    with pytest.raises(ValueError, match="verified but not replayed"):
        replay_artifact(path)


def test_artifact_source_must_match_installed_source_by_default(tmp_path) -> None:
    artifact = artifact_dict(run_benchmark(default_config()))
    artifact["provenance"]["fleetvla_source_sha256"] = "0" * 64
    artifact.pop("sha256")
    _seal_artifact(artifact)
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="does not match installed"):
        verify_artifact(path)
    assert verify_artifact(path, allow_source_mismatch=True)["sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact["config"].pop("model_id"), "model_id"),
        (
            lambda artifact: artifact["config"].update(execution_horizon=51),
            "execution_horizon",
        ),
        (
            lambda artifact: artifact["metrics"]["tasks"].clear(),
            "task metrics",
        ),
        (
            lambda artifact: artifact["metrics"]["system"].update(
                action_age_p95_s=0.01
            ),
            "action_age_p95_s",
        ),
        (lambda artifact: artifact.update(events=[]), "events"),
    ],
)
def test_system_artifact_rejects_invalid_kind_specific_schema(
    tmp_path, mutation, message
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    mutation(artifact)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match=message):
        verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact["metrics"]["system"].update(useful_actions=2),
        lambda artifact: _remove_event(artifact, "inference_completed"),
        lambda artifact: _remove_event(artifact, "inference_started"),
        lambda artifact: _event(artifact, "batch_dispatched")["details"].update(
            batch_size=0
        ),
    ],
)
def test_system_artifact_rejects_metrics_that_disagree_with_events(
    tmp_path, mutation
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    mutation(artifact)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError):
        verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: _event(artifact, "batch_dispatched")["details"].update(
            session_ids=["alien"]
        ),
        lambda artifact: _event(artifact, "batch_dispatched")["details"].update(
            session_ids=["libero_spatial-0", "libero_spatial-0"]
        ),
        lambda artifact: _event(artifact, "batch_dispatched")["details"][
            "selected_state"
        ][0].update(session_id="alien"),
        lambda artifact: _event(artifact, "batch_dispatched").update(
            kind="invented_event"
        ),
        lambda artifact: _event(artifact, "inference_completed")["details"].update(
            batch_size=0
        ),
        lambda artifact: _event(artifact, "inference_completed")["details"][
            "execution_horizons"
        ].__setitem__(0, 999),
        lambda artifact: _event(artifact, "inference_completed")["details"][
            "output_shapes"
        ][0].__setitem__(1, 999),
        lambda artifact: _event(artifact, "batch_dispatched")["details"].update(
            session_ids=[[]]
        ),
        lambda artifact: artifact["events"].append(
            {
                "time_s": 1.0,
                "kind": "session_registered",
                "session_id": "libero_spatial-0",
                "details": None,
            }
        ),
    ],
)
def test_system_artifact_rejects_invalid_event_details(tmp_path, mutation) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    mutation(artifact)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError):
        verify_artifact(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_replace_chunk_with_equal_horizon_rejection, "unequal action counts"),
        (_forge_action_age, "action outcome"),
        (_forge_inference_latency, "inference timing"),
        (_forge_backend_latency, "inference timing"),
        (_forge_chunk_age, "accepted chunk"),
        (_replace_chunk_with_wrong_expected_horizon, "system config"),
        (
            lambda artifact: _event(artifact, "batch_dispatched")["details"][
                "selected_state"
            ][0].update(buffer_horizon_s=999),
            "selected_state",
        ),
        (
            lambda artifact: _event(artifact, "batch_dispatched")["details"][
                "selected_state"
            ][0].update(request_age_s=999),
            "selected_state",
        ),
    ],
)
def test_system_artifact_rejects_internally_inconsistent_values(
    tmp_path, mutation, message
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    mutation(artifact)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match=message):
        verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: _remove_event(artifact, "action_dequeued"),
        lambda artifact: _event(artifact, "action_executed")["details"].update(
            sequence=999
        ),
        lambda artifact: _event(artifact, "action_dequeued")["details"].update(
            remaining_steps=6
        ),
        lambda artifact: artifact["config"].update(duration_s=2.0),
    ],
)
def test_system_artifact_rejects_invalid_session_lifecycle(
    tmp_path, mutation
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    mutation(artifact)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError):
        verify_artifact(path)


def test_system_artifact_binds_action_sequence_to_observation(tmp_path) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    for kind in (
        "request_dispatched",
        "chunk_accepted",
        "action_dequeued",
        "action_executed",
    ):
        _event(artifact, kind)["details"]["sequence"] = 7
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="request does not match"):
        verify_artifact(path)


def test_system_artifact_accepts_failure_reset_and_fallback(tmp_path) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    session_id = "libero_spatial-0"
    completion_index = artifact["events"].index(
        _event(artifact, "inference_completed")
    )
    artifact["events"][completion_index:] = [
        {
            "time_s": 0.052,
            "kind": "inference_failed",
            "session_id": None,
            "details": {
                "error": "test failure",
                "session_ids": [session_id],
                "phase": "inference",
            },
        },
        {
            "time_s": 0.053,
            "kind": "session_reset",
            "session_id": session_id,
            "details": {"generation": 1},
        },
        {
            "time_s": 0.054,
            "kind": "endpoint_fallback",
            "session_id": session_id,
            "details": None,
        },
        {
            "time_s": 0.055,
            "kind": "endpoint_task_step",
            "session_id": session_id,
            "details": {
                "reward": 0.0,
                "success": False,
                "terminated": False,
                "truncated": False,
            },
        },
        {
            "time_s": 1.0,
            "kind": "action_starved",
            "session_id": session_id,
            "details": None,
        },
        {
            "time_s": 1.0,
            "kind": "endpoint_fallback",
            "session_id": session_id,
            "details": None,
        },
        {
            "time_s": 1.0,
            "kind": "endpoint_task_step",
            "session_id": session_id,
            "details": {
                "reward": 0.0,
                "success": False,
                "terminated": False,
                "truncated": False,
            },
        },
    ]
    artifact["metrics"]["system"].update(
        useful_actions=0,
        starvation_frequency=1.0,
        starvation_duration_s=1.0,
        action_age_p50_s=None,
        action_age_p95_s=None,
        backend_utilization=0.0,
    )
    artifact["metrics"]["system"]["per_session"][session_id].update(
        actions=0,
        starved_ticks=1,
        starvation_duration_s=1.0,
        useful_progress_ratio=0.0,
    )
    artifact["metrics"]["tasks"][session_id].update(
        reward=0.0,
        steps=2,
        successes=0,
        episodes=0,
        terminated=0,
        truncated=0,
    )
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    assert verify_artifact(path)["artifact_kind"] == "system"

    artifact.pop("sha256")
    reset_index = artifact["events"].index(_event(artifact, "session_reset"))
    artifact["events"].insert(
        reset_index,
        {
            "time_s": 0.052,
            "kind": "action_starved",
            "session_id": session_id,
            "details": None,
        },
    )
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="failure must be followed"):
        verify_artifact(path)


def test_system_artifact_requires_stale_chunk_after_in_flight_reset(
    tmp_path,
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    session_id = "libero_spatial-0"
    accepted_index = artifact["events"].index(_event(artifact, "chunk_accepted"))
    artifact["events"][accepted_index + 1 : accepted_index + 1] = [
        {
            "time_s": 0.07,
            "kind": "observation_ready",
            "session_id": session_id,
            "details": {"sequence": 1},
        },
        {
            "time_s": 0.071,
            "kind": "request_dispatched",
            "session_id": session_id,
            "details": {"sequence": 1, "batch_size": 1},
        },
        {
            "time_s": 0.072,
            "kind": "batch_dispatched",
            "session_id": None,
            "details": {
                "batch_size": 1,
                "session_ids": [session_id],
                "reason": "test",
                "selected_state": [
                    {
                        "session_id": session_id,
                        "buffer_horizon_s": 8.0,
                        "request_age_s": 0.002,
                    }
                ],
            },
        },
        {
            "time_s": 0.073,
            "kind": "scheduler_cost_estimate",
            "session_id": None,
            "details": {
                "batch_size": 1,
                "base_latency_s": 0.03,
                "per_item_latency_s": 0.01,
                "estimated_latency_s": 0.04,
            },
        },
        {
            "time_s": 0.074,
            "kind": "inference_started",
            "session_id": None,
            "details": {"batch_size": 1},
        },
    ]
    artifact["events"].extend(
        [
            {
                "time_s": 0.21,
                "kind": "inference_completed",
                "session_id": None,
                "details": {
                    "batch_size": 1,
                    "latency_s": 0.136,
                    "backend_reported_latency_s": 0.136,
                    "output_shapes": [[1, 50, 7]],
                    "execution_horizons": [8],
                },
            },
            {
                "time_s": 0.22,
                "kind": "chunk_rejected_stale",
                "session_id": session_id,
                "details": None,
            },
        ]
    )
    artifact["metrics"]["system"].update(
        batch_sizes=[1, 1], backend_utilization=0.04 + 0.136
    )
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))
    assert verify_artifact(path)["artifact_kind"] == "system"

    artifact.pop("sha256")
    stale = _event(artifact, "chunk_rejected_stale")
    stale["kind"] = "chunk_accepted"
    stale["details"] = {"sequence": 1, "actions": 8, "action_age_s": 0.01}
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="session invalidation"):
        verify_artifact(path)


def test_system_artifact_requires_disconnect_after_action_failure(tmp_path) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    session_id = "libero_spatial-0"
    executed = _event(artifact, "action_executed")
    executed["kind"] = "action_rejected_endpoint"
    for kind in (
        "endpoint_task_step",
        "endpoint_episode_boundary",
        "session_reset",
    ):
        _remove_event(artifact, kind)
    artifact["events"].extend(
        [
            {
                "time_s": 0.11,
                "kind": "endpoint_action_failed",
                "session_id": session_id,
                "details": {"error": "test failure"},
            },
            {
                "time_s": 0.12,
                "kind": "session_disconnected",
                "session_id": session_id,
                "details": None,
            },
        ]
    )
    artifact["metrics"]["system"].update(
        useful_actions=0,
        starvation_frequency=1.0,
        starvation_duration_s=1.0,
        action_age_p50_s=None,
        action_age_p95_s=None,
    )
    artifact["metrics"]["system"]["per_session"][session_id].update(
        actions=0,
        starved_ticks=1,
        starvation_duration_s=1.0,
        useful_progress_ratio=0.0,
    )
    artifact["metrics"]["tasks"][session_id].update(
        reward=0.0,
        steps=0,
        successes=0,
        episodes=0,
        terminated=0,
        truncated=0,
    )
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))
    assert verify_artifact(path)["artifact_kind"] == "system"

    artifact.pop("sha256")
    _remove_event(artifact, "endpoint_action_failed")
    _remove_event(artifact, "session_disconnected")
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="incomplete session lifecycle"):
        verify_artifact(path)


@pytest.mark.parametrize(
    ("predecessor", "time_s", "message"),
    [
        ("request_dispatched", 0.006, "enter its batch"),
        ("action_dequeued", 0.1, "receive its outcome"),
    ],
)
def test_system_artifact_rejects_local_event_during_atomic_transition(
    tmp_path, predecessor, time_s, message
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    index = artifact["events"].index(_event(artifact, predecessor)) + 1
    artifact["events"].insert(
        index,
        {
            "time_s": time_s,
            "kind": "observation_suppressed_backpressure",
            "session_id": "libero_spatial-0",
            "details": None,
        },
    )
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match=message):
        verify_artifact(path)


def test_system_artifact_requires_boundary_before_reset(tmp_path) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    reset = _event(artifact, "session_reset")
    artifact["events"].remove(reset)
    boundary_index = artifact["events"].index(
        _event(artifact, "endpoint_episode_boundary")
    )
    artifact["events"].insert(boundary_index, reset)
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="followed by an episode boundary"):
        verify_artifact(path)


@pytest.mark.parametrize(
    "predecessor",
    ["endpoint_task_step", "endpoint_episode_boundary"],
)
def test_system_artifact_rejects_action_before_required_terminal_transition(
    tmp_path, predecessor
) -> None:
    artifact = _valid_system_artifact()
    artifact.pop("sha256")
    dequeue = json.loads(json.dumps(_event(artifact, "action_dequeued")))
    execute = json.loads(json.dumps(_event(artifact, "action_executed")))
    dequeue["details"].update(remaining_steps=6, action_index=1)
    execute["details"].update(action_index=1)
    preceding_event = _event(artifact, predecessor)
    dequeue["time_s"] = preceding_event["time_s"]
    execute["time_s"] = preceding_event["time_s"]
    index = artifact["events"].index(preceding_event) + 1
    artifact["events"][index:index] = [dequeue, execute]
    artifact["metrics"]["system"].update(
        useful_actions=2,
        starvation_frequency=0.0,
    )
    artifact["metrics"]["system"]["per_session"]["libero_spatial-0"].update(
        actions=2,
        useful_progress_ratio=1.0,
    )
    artifact["config"]["duration_s"] = 2.0
    _seal_artifact(artifact)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="followed by"):
        verify_artifact(path)


def test_algorithm_final_inference_must_finish_when_due(tmp_path) -> None:
    config = replace(default_config(), duration_s=0.01)
    path = write_artifact(run_benchmark(config), tmp_path / "run.json")
    assert verify_artifact(path)
    artifact = json.loads(path.read_text())
    artifact.pop("sha256")
    _event(artifact, "inference_started")["details"]["latency_s"] = 0.001
    artifact["metrics"]["backend_utilization"] = 0.1
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="due inference completion"):
        verify_artifact(path)


@pytest.mark.parametrize("kind", ["algorithm", "system"])
@pytest.mark.parametrize("mutation", ["time", "session"])
def test_artifact_rejects_invalid_event_stream(
    tmp_path, kind, mutation
) -> None:
    if kind == "algorithm":
        path = write_artifact(run_benchmark(default_config()), tmp_path / "run.json")
        artifact = json.loads(path.read_text())
    else:
        path = tmp_path / "system.json"
        artifact = _valid_system_artifact()
    artifact.pop("sha256")
    if mutation == "time":
        if kind == "system":
            artifact["events"][0]["time_s"] = 1.0
        artifact["events"].append(
            {
                "time_s": 0.0,
                "kind": "invalid_order",
                "session_id": None,
                "details": None,
            }
        )
    else:
        artifact["events"][0]["session_id"] = "alien"
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="event"):
        verify_artifact(path)


def test_artifact_rejects_non_object_root_and_bad_nested_config(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="root"):
        verify_artifact(path)

    artifact = artifact_dict(run_benchmark(default_config()))
    artifact["config"]["robots"] = ["not-an-object"]
    artifact.pop("sha256")
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="config"):
        verify_artifact(path)


def test_local_artifact_cannot_omit_embedded_source(tmp_path) -> None:
    scheduler_path = tmp_path / "custom.py"
    scheduler_path.write_text(
        "from fleetvla import ScheduleDecision\n"
        "class Custom:\n"
        "    def schedule(self, fleet, costs):\n"
        "        return ScheduleDecision(tuple(s.session_id for s in "
        "fleet.ready_sessions[:fleet.max_batch_size]))\n"
    )
    config = replace(default_config(), scheduler=f"{scheduler_path}:Custom")
    path = write_artifact(run_benchmark(config), tmp_path / "run.json")
    artifact = json.loads(path.read_text())
    artifact["provenance"]["scheduler_source"] = None
    artifact["provenance"]["scheduler_source_sha256"] = None
    artifact.pop("sha256")
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="must embed"):
        verify_artifact(path)
    with pytest.raises(ValueError, match="must embed"):
        replay_artifact(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_version", True, "version"),
        ("config", [], "config"),
        ("metrics", "not metrics", "metrics"),
        ("events", None, "events"),
    ],
)
def test_artifact_verifier_rejects_invalid_schema(
    tmp_path, field, value, message
) -> None:
    path = write_artifact(run_benchmark(default_config()), tmp_path / "run.json")
    artifact = json.loads(path.read_text())
    artifact[field] = value
    artifact.pop("sha256")
    _seal_artifact(artifact)
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match=message):
        verify_artifact(path)


def test_modified_artifact_is_rejected(tmp_path) -> None:
    path = write_artifact(run_benchmark(default_config()), tmp_path / "run.json")
    data = json.loads(path.read_text())
    data["metrics"]["useful_actions"] += 1
    path.write_text(json.dumps(data))

    try:
        load_artifact(path)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("modified benchmark artifact was accepted")


def test_trace_environment_reuses_workload_with_another_scheduler(tmp_path) -> None:
    source_run = run_benchmark(default_config())
    source = write_artifact(source_run, tmp_path / "source.json")
    config = trace_config(source, "edf")

    rerun = run_benchmark(config)

    assert rerun.config.environment == "trace"
    assert rerun.config.scheduler == "edf"
    assert rerun.config.trace_source == str(source)
    assert rerun.metrics.useful_actions > 0
    source_observations = source_run.result.count("observation_ready")
    replayed_arrivals = rerun.result.count("observation_ready") + rerun.result.count(
        "observation_dropped_backpressure"
    )
    assert replayed_arrivals == source_observations

    derived = write_artifact(rerun, tmp_path / "derived.json")
    source.unlink()
    _, matches = replay_artifact(derived)
    assert matches


def test_contended_workload_exposes_scheduler_tradeoffs() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "contention.json"
    base = load_config(path)
    runs = {
        name: run_benchmark(replace(base, scheduler=name)).metrics
        for name in ("fifo", "round-robin", "edf", "adaptive-slack")
    }

    signatures = {
        (
            metrics.useful_actions,
            round(metrics.fairness, 6),
            tuple(
                int(session["actions"])
                for session in metrics.per_session.values()
            ),
        )
        for metrics in runs.values()
    }
    assert len(signatures) >= 3


def test_local_scheduler_artifact_embeds_exact_portable_source(tmp_path) -> None:
    scheduler_path = tmp_path / "custom.py"
    scheduler_path.write_text(
        "from fleetvla import ScheduleDecision\n"
        "class Custom:\n"
        "    def schedule(self, fleet, costs):\n"
        "        del costs\n"
        "        if not fleet.ready_sessions:\n"
        "            return ScheduleDecision((), 'no work')\n"
        "        oldest = min(s.request_time_s for s in "
        "fleet.ready_sessions)\n"
        "        if fleet.now_s < oldest + 0.001:\n"
        "            return ScheduleDecision((), 'wait', "
        "defer_until_s=oldest + 0.001)\n"
        "        return ScheduleDecision(tuple(s.session_id for s in "
        "fleet.ready_sessions[:fleet.max_batch_size]), 'embedded')\n"
    )
    config = replace(
        default_config(), scheduler=f"{scheduler_path}:Custom"
    )
    artifact = write_artifact(run_benchmark(config), tmp_path / "run.json")
    assert any(
        event["kind"] == "dispatch_deferred"
        for event in load_artifact(artifact)["events"]
    )
    moved = tmp_path / "shared" / "run.json"
    moved.parent.mkdir()
    shutil.move(artifact, moved)
    scheduler_path.write_text("raise RuntimeError('current source must not run')\n")

    try:
        replay_artifact(moved)
    except ValueError as error:
        assert "--allow-embedded-scheduler" in str(error)
    else:
        raise AssertionError("embedded code executed without explicit consent")

    _, matches = replay_artifact(moved, allow_embedded_scheduler=True)
    assert matches


def test_matrix_output_sanitizes_local_scheduler_filename(tmp_path) -> None:
    scheduler_path = tmp_path / "custom scheduler.py"
    scheduler_path.write_text(
        "from fleetvla import ScheduleDecision\n"
        "class Custom:\n"
        "    def schedule(self, fleet, costs):\n"
        "        return ScheduleDecision(tuple(s.session_id for s in "
        "fleet.ready_sessions[:fleet.max_batch_size]))\n"
    )
    output = tmp_path / "results"

    assert main(
        [
            "benchmark",
            "--duration",
            "0.1",
            "--scheduler",
            "fifo",
            "--scheduler",
            f"{scheduler_path}:Custom",
            "--output",
            str(output),
        ]
    ) == 0

    names = {path.name for path in output.iterdir()}
    assert "heterogeneous-two-arm-fifo-s0.json" in names
    local_names = names - {"heterogeneous-two-arm-fifo-s0.json"}
    assert len(local_names) == 1
    assert next(iter(local_names)).startswith(
        "heterogeneous-two-arm-custom-scheduler-Custom-"
    )


def test_benchmark_and_trace_times_must_be_finite() -> None:
    from fleetvla.benchmark import BenchmarkConfig

    data = default_config().as_dict()
    data["duration_s"] = math.nan
    try:
        BenchmarkConfig.from_dict(data)
    except ValueError as error:
        assert "duration" in str(error)
    else:
        raise AssertionError("NaN duration was accepted")

    data = default_config().as_dict()
    data["environment"] = "trace"
    data["observation_schedule"] = [[math.nan, "fast-arm"]]
    try:
        BenchmarkConfig.from_dict(data)
    except ValueError as error:
        assert "observation times" in str(error)
    else:
        raise AssertionError("NaN trace time was accepted")

    for value in (math.nan, math.inf, -math.inf):
        data = default_config().as_dict()
        data["environment"] = "trace"
        data["observation_schedule"] = [[value, "fast-arm"]]
        with pytest.raises(ValueError, match="observation times"):
            BenchmarkConfig.from_dict(data)

        with pytest.raises(ValueError, match="observation times"):
            replace(
                default_config(),
                environment="trace",
                observation_schedule=((value, "fast-arm"),),
            )
