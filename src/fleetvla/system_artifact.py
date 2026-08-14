"""Validation specific to measured wall-clock system artifacts."""

from __future__ import annotations

from typing import Any

from .artifact_lifecycle import (
    _validate_inference_lifecycle,
    _validate_system_session_lifecycle,
)
from .artifact_metrics import (
    _validate_benchmark_metrics,
    _validate_metrics_match_events,
)
from .artifact_schema import (
    _finite_number,
    _non_negative_integer,
    _positive_integer,
    _require_keys,
    _require_strings,
    _validate_events,
)
from .runtime import ACTION_EXECUTION_POLICIES
from .simulation import RobotSpec


def _validate_system_config(artifact: dict[str, Any]) -> set[str]:
    provenance = artifact["provenance"]
    _require_strings(
        provenance,
        "fleetvla_version",
        "lerobot_version",
        "libero_version",
        "python_version",
        "platform",
        "torch_version",
        "accelerator",
    )
    cuda_version = provenance.get("cuda_version")
    if cuda_version is not None and not isinstance(cuda_version, str):
        raise ValueError("system provenance cuda_version must be a string or null")

    config = artifact["config"]
    _require_keys(
        config,
        "clock",
        "environment",
        "suite",
        "task_ids",
        "task_descriptions",
        "model_id",
        "model_revision",
        "scheduler",
        "scheduler_config",
        "duration_s",
        "max_batch_size",
        "control_hz",
        "model_output_horizon",
        "execution_horizon",
        "action_execution",
        "episode_length",
        "seed",
    )
    if config["clock"] != "monotonic-wall" or config["environment"] != "libero":
        raise ValueError("system artifact must describe a wall-clock LIBERO run")
    _require_strings(config, "suite", "model_id", "scheduler")
    if config["model_revision"] is not None and not isinstance(
        config["model_revision"], str
    ):
        raise ValueError("model_revision must be a string or null")
    if not isinstance(config["scheduler_config"], dict):
        raise ValueError("scheduler_config must be an object")
    for timeout_field in (
        "scheduler_timeout_s",
        "inference_timeout_s",
        "endpoint_timeout_s",
    ):
        if timeout_field not in config:
            continue
        _finite_number(
            config[timeout_field],
            timeout_field,
            minimum=0,
            strict=True,
        )
    task_ids = config["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(type(task_id) is not int or task_id < 0 for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("system task_ids must be unique non-negative integers")
    expected_sessions = {f"{config['suite']}-{task_id}" for task_id in task_ids}
    descriptions = config["task_descriptions"]
    if not isinstance(descriptions, dict) or set(descriptions) != expected_sessions:
        raise ValueError("task_descriptions must match the configured task sessions")
    if any(not isinstance(value, str) or not value for value in descriptions.values()):
        raise ValueError("task descriptions must be non-empty strings")
    _finite_number(config["duration_s"], "duration_s", minimum=0, strict=True)
    _finite_number(config["control_hz"], "control_hz", minimum=0, strict=True)
    for field in (
        "max_batch_size",
        "model_output_horizon",
        "execution_horizon",
        "episode_length",
    ):
        _positive_integer(config[field], field)
    if config["execution_horizon"] > config["model_output_horizon"]:
        raise ValueError("execution_horizon cannot exceed model_output_horizon")
    if config["action_execution"] not in ACTION_EXECUTION_POLICIES:
        raise ValueError("unsupported system action execution policy")
    if type(config["seed"]) is not int:
        raise ValueError("system seed must be an integer")

    metrics = artifact["metrics"]
    _require_keys(metrics, "system", "tasks")
    _validate_benchmark_metrics(metrics["system"], expected_sessions)
    tasks = metrics["tasks"]
    if not isinstance(tasks, dict) or set(tasks) != expected_sessions:
        raise ValueError("task metrics must match the configured task sessions")
    for task_metrics in tasks.values():
        if not isinstance(task_metrics, dict):
            raise ValueError("task metrics must be objects")
        _require_keys(
            task_metrics,
            "reward",
            "steps",
            "successes",
            "episodes",
            "terminated",
            "truncated",
        )
        _finite_number(task_metrics["reward"], "task reward")
        for field in ("steps", "successes", "episodes", "terminated", "truncated"):
            _non_negative_integer(task_metrics[field], f"task {field}")
        if task_metrics["successes"] > task_metrics["episodes"]:
            raise ValueError("task successes cannot exceed episodes")
        outcomes = task_metrics["terminated"] + task_metrics["truncated"]
        if not task_metrics["episodes"] <= outcomes <= 2 * task_metrics["episodes"]:
            raise ValueError("task episode outcomes must account for every episode")
    return expected_sessions


def _validate_task_event_counts(artifact: dict[str, Any]) -> None:
    events = artifact["events"]
    for session_id, task_metrics in artifact["metrics"]["tasks"].items():
        task_events = [
            event
            for event in events
            if event["kind"] == "endpoint_task_step"
            and event["session_id"] == session_id
        ]
        steps = len(task_events)
        episodes = sum(
            event["kind"] == "endpoint_episode_boundary"
            and event["session_id"] == session_id
            for event in events
        )
        event_metrics = {
            "reward": sum(event["details"]["reward"] for event in task_events),
            "steps": steps,
            "successes": sum(event["details"]["success"] for event in task_events),
            "episodes": episodes,
            "terminated": sum(event["details"]["terminated"] for event in task_events),
            "truncated": sum(event["details"]["truncated"] for event in task_events),
        }
        if task_metrics != event_metrics:
            raise ValueError("task metrics do not match system events")


def validate_system_artifact(artifact: dict[str, Any]) -> None:
    session_ids = _validate_system_config(artifact)
    config = artifact["config"]
    _validate_events(artifact["events"], session_ids)
    _validate_inference_lifecycle(
        artifact["events"],
        system=True,
        expected_sessions=session_ids,
        duration_s=config["duration_s"],
        max_batch_size=config["max_batch_size"],
        model_output_horizon=config["model_output_horizon"],
        execution_horizon=config["execution_horizon"],
        require_scheduler_events=("scheduler_timeout_s" in config),
    )
    _validate_system_session_lifecycle(
        artifact["events"],
        session_ids,
        duration_s=config["duration_s"],
        control_hz=config["control_hz"],
        action_execution=config["action_execution"],
    )
    robots = tuple(
        RobotSpec(
            session_id,
            control_hz=config["control_hz"],
            chunk_size=config["execution_horizon"],
        )
        for session_id in sorted(session_ids)
    )
    _validate_metrics_match_events(
        artifact["metrics"]["system"],
        artifact["events"],
        robots,
        config["duration_s"],
    )
    _validate_task_event_counts(artifact)
