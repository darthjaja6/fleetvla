"""Validation specific to deterministic algorithm artifacts."""

from __future__ import annotations

from typing import Any

from .artifact_lifecycle import _validate_inference_lifecycle
from .artifact_metrics import (
    _validate_benchmark_metrics,
    _validate_metrics_match_events,
)
from .artifact_schema import _require_strings, _validate_events
from .benchmark import BenchmarkConfig


def validate_algorithm_artifact(
    artifact: dict[str, Any], provenance: dict[str, Any]
) -> None:
    try:
        config = BenchmarkConfig.from_dict(artifact["config"])
    except (KeyError, TypeError) as error:
        raise ValueError("algorithm artifact config is invalid") from error
    session_ids = {robot.session_id for robot in config.robots}
    _validate_events(artifact["events"], session_ids)
    _validate_inference_lifecycle(
        artifact["events"],
        system=False,
        expected_sessions=session_ids,
        duration_s=config.duration_s,
        max_batch_size=config.max_batch_size,
    )
    _validate_benchmark_metrics(artifact["metrics"], session_ids)
    _validate_metrics_match_events(
        artifact["metrics"], artifact["events"], config.robots, config.duration_s
    )
    _require_strings(provenance, "fleetvla_version", "python_version", "platform")
