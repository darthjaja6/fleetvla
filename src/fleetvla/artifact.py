"""Artifact I/O, top-level validation, and deterministic replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .algorithm_artifact import validate_algorithm_artifact
from .artifact_schema import _is_sha256
from .benchmark import (
    ARTIFACT_VERSION,
    BenchmarkConfig,
    BenchmarkRun,
    _event_dict,
    artifact_dict,
    fleetvla_source_sha256,
    run_benchmark,
)
from .system_artifact import validate_system_artifact


def write_artifact(run: BenchmarkRun, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def verify_artifact(
    path: str | Path, *, allow_source_mismatch: bool = False
) -> dict[str, Any]:
    """Validate artifact kind, schema, checksum, and embedded source hash."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("artifact root must be an object")
    version = artifact.get("artifact_version")
    if type(version) is not int or version != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact version")
    kind = artifact.get("artifact_kind", "algorithm")
    if kind not in {"algorithm", "system"}:
        raise ValueError(f"unsupported artifact kind: {kind!r}")
    required = {"provenance", "config", "metrics", "events", "sha256"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"artifact is missing fields: {', '.join(sorted(missing))}")
    provenance = artifact["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("artifact provenance must be an object")
    if not isinstance(artifact["config"], dict):
        raise ValueError("artifact config must be an object")
    if not isinstance(artifact["metrics"], dict):
        raise ValueError("artifact metrics must be an object")
    if not isinstance(artifact["events"], list):
        raise ValueError("artifact events must be an array")
    source_identity = provenance.get("fleetvla_source_sha256")
    if not _is_sha256(source_identity):
        raise ValueError("artifact lacks an exact FleetVLA source identifier")
    if not allow_source_mismatch and source_identity != fleetvla_source_sha256():
        raise ValueError("artifact FleetVLA source does not match installed source")
    claimed = artifact.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual:
        raise ValueError("artifact checksum does not match its contents")
    source = provenance.get("scheduler_source")
    source_digest = provenance.get("scheduler_source_sha256")
    if (source is None) != (source_digest is None):
        raise ValueError("embedded scheduler source and checksum must appear together")
    scheduler_spec = artifact["config"].get("scheduler")
    if not isinstance(scheduler_spec, str) or not scheduler_spec:
        raise ValueError("artifact scheduler must be a non-empty string")
    if ":" in scheduler_spec and source is None:
        raise ValueError("local scheduler artifacts must embed scheduler source")
    if source is not None:
        if not isinstance(source, str) or not _is_sha256(source_digest):
            raise ValueError("embedded scheduler source checksum is invalid")
        actual_source_digest = hashlib.sha256(source.encode()).hexdigest()
        if actual_source_digest != source_digest:
            raise ValueError("embedded scheduler checksum does not match its source")
    if kind == "algorithm":
        validate_algorithm_artifact(artifact, provenance)
    else:
        validate_system_artifact(artifact)
    artifact["sha256"] = claimed
    return artifact


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = verify_artifact(path)
    if artifact.get("artifact_kind", "algorithm") != "algorithm":
        raise ValueError("system artifacts can be verified but not replayed")
    return artifact


def replay_artifact(
    path: str | Path, *, allow_embedded_scheduler: bool = False
) -> tuple[BenchmarkRun, bool]:
    artifact = load_artifact(path)
    config = BenchmarkConfig.from_dict(artifact["config"])
    source = artifact.get("provenance", {}).get("scheduler_source")
    if ":" in config.scheduler and source is None:
        raise ValueError("local scheduler artifact is missing embedded source")
    if source is None:
        replayed = run_benchmark(config)
    else:
        if not allow_embedded_scheduler:
            raise ValueError(
                "artifact contains executable scheduler code; pass "
                "--allow-embedded-scheduler after reviewing its source"
            )
        class_name = config.scheduler.rsplit(":", 1)[1]
        with tempfile.TemporaryDirectory(prefix="fleetvla-replay-") as directory:
            scheduler_path = Path(directory) / "embedded_scheduler.py"
            scheduler_path.write_text(source, encoding="utf-8")
            replay_config = replace(config, scheduler=f"{scheduler_path}:{class_name}")
            replayed_result = run_benchmark(replay_config)
        replayed = replace(replayed_result, config=config)
    matches = (
        replayed.metrics.as_dict() == artifact["metrics"]
        and [_event_dict(event) for event in replayed.result.events]
        == artifact["events"]
    )
    return replayed, matches
