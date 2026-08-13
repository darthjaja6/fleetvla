"""Reproducible benchmark configuration, metrics, artifacts, and replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .backend import SyntheticBackend
from .schedulers import create_scheduler
from .simulation import FleetSimulator, RobotSpec, SimulationResult
from .trace import Event
from .types import InferenceCostModel


ARTIFACT_VERSION = 1
_EVENT_CLOCK_TOLERANCE_S = 0.01


def fleetvla_source_sha256() -> str:
    """Identify the installed FleetVLA Python implementation exactly."""

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    scheduler: str
    scheduler_config: dict[str, Any]
    environment: str
    backend: str
    scenario: str
    robots: tuple[RobotSpec, ...]
    duration_s: float
    seed: int
    base_latency_s: float
    per_item_latency_s: float
    max_batch_size: int
    action_execution: str = "sequential-buffer"
    trace_source: str | None = None
    observation_schedule: tuple[tuple[float, str], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler, str) or not self.scheduler:
            raise ValueError("scheduler must be a non-empty string")
        if not isinstance(self.scheduler_config, dict):
            raise ValueError("scheduler_config must be an object")
        if self.environment not in {"synthetic", "trace"}:
            raise ValueError(f"unsupported environment: {self.environment}")
        if self.backend != "synthetic":
            raise ValueError(f"unsupported backend: {self.backend}")
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a non-empty string")
        if not self.robots:
            raise ValueError("at least one robot is required")
        if len({robot.session_id for robot in self.robots}) != len(self.robots):
            raise ValueError("robot session_id values must be unique")
        for robot in self.robots:
            robot.session_config()
        if self.action_execution != "sequential-buffer":
            raise ValueError(
                f"unsupported action execution policy: {self.action_execution}"
            )
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration must be finite and positive")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        InferenceCostModel(self.base_latency_s, self.per_item_latency_s)
        if type(self.max_batch_size) is not int or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if self.observation_schedule is not None:
            for time_s, session_id in self.observation_schedule:
                _finite_time(time_s)
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("trace session IDs must be non-empty strings")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["robots"] = [asdict(robot) for robot in self.robots]
        if self.observation_schedule is not None:
            data["observation_schedule"] = [
                [time_s, session_id]
                for time_s, session_id in self.observation_schedule
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        unknown = set(data) - {
            "scheduler",
            "scheduler_config",
            "environment",
            "backend",
            "scenario",
            "robots",
            "duration_s",
            "seed",
            "base_latency_s",
            "per_item_latency_s",
            "max_batch_size",
            "action_execution",
            "trace_source",
            "observation_schedule",
        }
        if unknown:
            raise ValueError(f"unknown benchmark fields: {', '.join(sorted(unknown))}")
        values = dict(data)
        values["robots"] = tuple(RobotSpec(**robot) for robot in data["robots"])
        if values.get("observation_schedule") is not None:
            values["observation_schedule"] = tuple(
                (_finite_time(time_s), str(session_id))
                for time_s, session_id in values["observation_schedule"]
            )
        values.setdefault("scheduler_config", {})
        return cls(**values)


def default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        scheduler="adaptive-slack",
        scheduler_config={},
        environment="synthetic",
        backend="synthetic",
        scenario="heterogeneous-two-arm",
        robots=(
            RobotSpec(
                "fast-arm",
                control_hz=20,
                chunk_size=4,
                request_threshold_s=0.1,
                network_latency_s=0.005,
                latency_budget_s=0.12,
            ),
            RobotSpec(
                "slow-arm",
                control_hz=10,
                chunk_size=4,
                request_threshold_s=0.15,
                network_latency_s=0.02,
                latency_budget_s=0.25,
            ),
        ),
        duration_s=3.0,
        seed=0,
        base_latency_s=0.03,
        per_item_latency_s=0.008,
        max_batch_size=2,
    )


def _finite_time(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("observation times must be finite and non-negative")
    return value


def load_config(path: str | Path) -> BenchmarkConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.pop("version", None)
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported benchmark config version {version!r}; expected {ARTIFACT_VERSION}"
        )
    return BenchmarkConfig.from_dict(data)


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    useful_actions: int
    starvation_frequency: float
    starvation_duration_s: float
    action_age_p50_s: float | None
    action_age_p95_s: float | None
    fairness: float
    batch_sizes: tuple[int, ...]
    backend_utilization: float
    per_session: dict[str, dict[str, float | int]]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batch_sizes"] = list(self.batch_sizes)
        return data


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    config: BenchmarkConfig
    result: SimulationResult
    metrics: BenchmarkMetrics


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    scheduler = create_scheduler(config.scheduler, config.scheduler_config)
    backend = SyntheticBackend(
        chunk_sizes={robot.session_id: robot.chunk_size for robot in config.robots},
        base_latency_s=config.base_latency_s,
        per_item_latency_s=config.per_item_latency_s,
    )
    observation_schedule = config.observation_schedule
    if config.environment == "trace":
        if observation_schedule is None:
            raise ValueError("trace environment requires an observation schedule")
    result = FleetSimulator(
        list(config.robots),
        backend=backend,
        scheduler=scheduler,
        max_batch_size=config.max_batch_size,
        observation_schedule=observation_schedule,
    ).run(config.duration_s)
    return BenchmarkRun(config, result, compute_metrics(result, config.robots))


def run_matrix(configs: Iterable[BenchmarkConfig]) -> tuple[BenchmarkRun, ...]:
    return tuple(run_benchmark(config) for config in configs)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def compute_metrics(
    result: SimulationResult, robots: tuple[RobotSpec, ...] | list[RobotSpec]
) -> BenchmarkMetrics:
    per_session: dict[str, dict[str, float | int]] = {}
    progress_ratios: list[float] = []
    total_starved = 0
    total_ticks = 0
    for robot in robots:
        executed = result.count("action_executed", robot.session_id)
        rejected = result.count("action_rejected_endpoint", robot.session_id)
        missed = sum(
            int(event.details["count"])
            for event in result.events
            if event.kind == "control_ticks_missed"
            and event.session_id == robot.session_id
            and event.details
        )
        starved = (
            result.count("action_starved", robot.session_id) + missed + rejected
        )
        ticks = executed + starved
        starvation_duration_s = starved / robot.control_hz
        ratio = executed / ticks if ticks else 0.0
        progress_ratios.append(ratio)
        total_starved += starved
        total_ticks += ticks
        per_session[robot.session_id] = {
            "actions": executed,
            "starved_ticks": starved,
            "starvation_duration_s": starvation_duration_s,
            "useful_progress_ratio": ratio,
        }
    fairness_denominator = len(progress_ratios) * sum(
        value * value for value in progress_ratios
    )
    fairness = (
        sum(progress_ratios) ** 2 / fairness_denominator
        if fairness_denominator
        else 1.0
    )
    action_ages = [
        float(event.details["action_age_s"])
        for event in result.events
        if event.kind == "action_executed" and event.details
    ]
    batch_sizes = tuple(
        int(event.details["batch_size"])
        for event in result.events
        if event.kind == "batch_dispatched" and event.details
    )
    modeled_inference_time = sum(
        float(event.details["latency_s"])
        for event in result.events
        if event.kind == "inference_started"
        and event.details
        and "latency_s" in event.details
    )
    measured_inference_time = sum(
        float(event.details["latency_s"])
        for event in result.events
        if event.kind == "inference_completed"
        and event.details
        and "latency_s" in event.details
    )
    return BenchmarkMetrics(
        useful_actions=sum(
            int(metrics["actions"]) for metrics in per_session.values()
        ),
        starvation_frequency=total_starved / total_ticks if total_ticks else 0.0,
        starvation_duration_s=sum(
            float(metrics["starvation_duration_s"])
            for metrics in per_session.values()
        ),
        action_age_p50_s=_percentile(action_ages, 0.5),
        action_age_p95_s=_percentile(action_ages, 0.95),
        fairness=fairness,
        batch_sizes=batch_sizes,
        backend_utilization=min(
            1.0,
            (measured_inference_time or modeled_inference_time) / result.duration_s,
        ),
        per_session=per_session,
    )


def _event_dict(event: Event) -> dict[str, Any]:
    # Normalize tuples and other JSON-compatible containers exactly as they are
    # persisted so an in-memory replay compares against the stored artifact.
    return json.loads(json.dumps(event.as_dict()))


def artifact_dict(run: BenchmarkRun) -> dict[str, Any]:
    scheduler_source_sha256 = None
    scheduler_source = None
    if ":" in run.config.scheduler:
        scheduler_path = Path(run.config.scheduler.rsplit(":", 1)[0]).resolve()
        scheduler_source = scheduler_path.read_text(encoding="utf-8")
        scheduler_source_sha256 = hashlib.sha256(
            scheduler_source.encode()
        ).hexdigest()
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "provenance": {
            "fleetvla_version": importlib.metadata.version("fleetvla"),
            "fleetvla_source_sha256": fleetvla_source_sha256(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "scheduler_source_sha256": scheduler_source_sha256,
            "scheduler_source": scheduler_source,
        },
        "config": run.config.as_dict(),
        "metrics": run.metrics.as_dict(),
        "events": [_event_dict(event) for event in run.result.events],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "sha256": digest}


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
        _validate_benchmark_metrics(
            artifact["metrics"],
            session_ids,
        )
        _validate_metrics_match_events(
            artifact["metrics"],
            artifact["events"],
            config.robots,
            config.duration_s,
        )
        _require_strings(
            provenance,
            "fleetvla_version",
            "python_version",
            "platform",
        )
    else:
        session_ids = _validate_system_artifact(artifact)
        _validate_events(artifact["events"], session_ids)
        _validate_inference_lifecycle(
            artifact["events"],
            system=True,
            expected_sessions=session_ids,
            duration_s=artifact["config"]["duration_s"],
            max_batch_size=artifact["config"]["max_batch_size"],
            model_output_horizon=artifact["config"]["model_output_horizon"],
            execution_horizon=artifact["config"]["execution_horizon"],
        )
        _validate_system_session_lifecycle(
            artifact["events"],
            session_ids,
            duration_s=artifact["config"]["duration_s"],
            control_hz=artifact["config"]["control_hz"],
        )
        system_robots = tuple(
            RobotSpec(
                session_id,
                control_hz=artifact["config"]["control_hz"],
                chunk_size=artifact["config"]["execution_horizon"],
            )
            for session_id in sorted(session_ids)
        )
        _validate_metrics_match_events(
            artifact["metrics"]["system"],
            artifact["events"],
            system_robots,
            artifact["config"]["duration_s"],
        )
        _validate_task_event_counts(artifact)
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
            replay_config = replace(
                config, scheduler=f"{scheduler_path}:{class_name}"
            )
            replayed_result = run_benchmark(replay_config)
        replayed = BenchmarkRun(config, replayed_result.result, replayed_result.metrics)
    matches = (
        replayed.metrics.as_dict() == artifact["metrics"]
        and [_event_dict(event) for event in replayed.result.events]
        == artifact["events"]
    )
    return replayed, matches


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_events(events: list[Any], expected_sessions: set[str]) -> None:
    if not events:
        raise ValueError("artifact events must not be empty")
    previous_time = -math.inf
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("artifact events must contain objects")
        _require_keys(event, "time_s", "kind", "session_id", "details")
        _finite_number(event["time_s"], "event time", minimum=0)
        if event["time_s"] < previous_time:
            raise ValueError("artifact event times must be non-decreasing")
        previous_time = event["time_s"]
        if not isinstance(event["kind"], str) or not event["kind"]:
            raise ValueError("event kind must be a non-empty string")
        if event["session_id"] is not None and not isinstance(
            event["session_id"], str
        ):
            raise ValueError("event session_id must be a string or null")
        if (
            event["session_id"] is not None
            and event["session_id"] not in expected_sessions
        ):
            raise ValueError("event session_id is not a configured session")
        if event["details"] is not None and not isinstance(event["details"], dict):
            raise ValueError("event details must be an object or null")
        _validate_event_contract(event, expected_sessions)


def _validate_event_contract(
    event: dict[str, Any], expected_sessions: set[str]
) -> None:
    kind = event["kind"]
    global_kinds = {
        "batch_dispatched",
        "inference_started",
        "inference_completed",
        "scheduler_cost_estimate",
        "inference_failed",
    }
    no_detail_kinds = {
        "session_registered",
        "action_starved",
        "chunk_rejected_stale",
        "chunk_rejected_unexpected",
        "action_ack_rejected_stale",
        "action_ack_rejected_duplicate",
        "action_ack_rejected_unexpected",
        "session_disconnected",
        "observation_dropped_backpressure",
        "observation_suppressed_backpressure",
        "endpoint_observation_unavailable",
        "endpoint_fallback",
        "endpoint_episode_boundary",
    }
    known_kinds = global_kinds | no_detail_kinds | {
        "observation_ready",
        "request_dispatched",
        "chunk_rejected_horizon",
        "chunk_accepted",
        "action_dequeued",
        "action_executed",
        "action_rejected_endpoint",
        "session_reconnected",
        "session_reset",
        "endpoint_observation_failed",
        "endpoint_fallback_failed",
        "endpoint_action_failed",
        "endpoint_close_failed",
        "control_ticks_missed",
        "endpoint_task_step",
    }
    if kind not in known_kinds:
        raise ValueError(f"unsupported artifact event kind: {kind!r}")
    if (kind in global_kinds) != (event["session_id"] is None):
        raise ValueError(f"event {kind} has an invalid session_id")
    if kind in no_detail_kinds:
        if event["details"] is not None:
            raise ValueError(f"event {kind} must not include details")
        return

    if kind == "batch_dispatched":
        details = _event_details(
            event, "batch_size", "session_ids", "reason", "selected_state"
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        session_ids = details["session_ids"]
        if (
            not isinstance(session_ids, list)
            or len(session_ids) != batch_size
            or any(
                not isinstance(session_id, str)
                or session_id not in expected_sessions
                for session_id in session_ids
            )
            or len(set(session_ids)) != batch_size
        ):
            raise ValueError("batch_dispatched session_ids are invalid")
        if not isinstance(details["reason"], str):
            raise ValueError("batch reason must be a string")
        selected = details["selected_state"]
        if not isinstance(selected, list) or len(selected) != batch_size:
            raise ValueError("selected_state must match the batch")
        for state, session_id in zip(selected, session_ids):
            if not isinstance(state, dict):
                raise ValueError("selected_state entries must be objects")
            _require_exact_keys(
                state, "session_id", "buffer_horizon_s", "request_age_s"
            )
            if state["session_id"] != session_id:
                raise ValueError("selected_state sessions must match the batch")
            _finite_number(state["buffer_horizon_s"], "buffer_horizon_s", minimum=0)
            _finite_number(state["request_age_s"], "request_age_s", minimum=0)
        return

    if kind == "inference_completed":
        details = _event_details(
            event,
            "batch_size",
            optional=(
                "latency_s",
                "backend_reported_latency_s",
                "output_shapes",
                "execution_horizons",
            ),
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        for field in ("latency_s", "backend_reported_latency_s"):
            if field in details:
                _finite_number(details[field], field, minimum=0)
        paired = ("output_shapes" in details, "execution_horizons" in details)
        if paired[0] != paired[1]:
            raise ValueError(
                "output shapes and execution horizons must appear together"
            )
        if paired[0]:
            shapes = details["output_shapes"]
            horizons = details["execution_horizons"]
            if (
                not isinstance(shapes, list)
                or len(shapes) != batch_size
                or not isinstance(horizons, list)
                or len(horizons) != batch_size
            ):
                raise ValueError("inference output metadata must match batch_size")
            for shape in shapes:
                if not isinstance(shape, list) or not shape:
                    raise ValueError("output shapes must be non-empty arrays")
                for size in shape:
                    _positive_integer(size, "output dimension")
                if shape[0] != batch_size:
                    raise ValueError("output batch dimension must match batch_size")
            for horizon in horizons:
                _positive_integer(horizon, "execution horizon")
        return

    if kind == "inference_started":
        details = _event_details(
            event, "batch_size", optional=("latency_s", "scheduler_reason")
        )
        _positive_integer(details["batch_size"], "batch_size")
        if "latency_s" in details:
            _finite_number(details["latency_s"], "latency_s", minimum=0)
        if "scheduler_reason" in details and not isinstance(
            details["scheduler_reason"], str
        ):
            raise ValueError("scheduler_reason must be a string")
        return

    if kind in {"scheduler_cost_estimate"}:
        details = _event_details(
            event,
            "batch_size",
            "base_latency_s",
            "per_item_latency_s",
            "estimated_latency_s",
        )
        batch_size = _positive_integer_value(details["batch_size"], "batch_size")
        for field in ("base_latency_s", "per_item_latency_s", "estimated_latency_s"):
            _finite_number(details[field], field, minimum=0)
        expected = (
            details["base_latency_s"]
            + details["per_item_latency_s"] * batch_size
        )
        if not math.isclose(details["estimated_latency_s"], expected):
            raise ValueError("estimated latency does not match the cost model")
        return

    if kind == "inference_failed":
        details = _event_details(event, "error", "session_ids", "phase")
        _error_string(details["error"])
        session_ids = details["session_ids"]
        if (
            not isinstance(session_ids, list)
            or not session_ids
            or any(
                not isinstance(session_id, str)
                or session_id not in expected_sessions
                for session_id in session_ids
            )
            or len(set(session_ids)) != len(session_ids)
        ):
            raise ValueError("inference_failed session_ids are invalid")
        if details["phase"] not in {"prepare", "inference"}:
            raise ValueError("inference failure phase is invalid")
        return

    details = _event_details(event, *_EVENT_REQUIRED_DETAILS[kind])
    if kind == "observation_ready":
        _non_negative_integer(details["sequence"], "sequence")
    elif kind == "request_dispatched":
        _non_negative_integer(details["sequence"], "sequence")
        _positive_integer(details["batch_size"], "batch_size")
    elif kind == "chunk_rejected_horizon":
        _positive_integer(details["expected_actions"], "expected_actions")
        _positive_integer(details["received_actions"], "received_actions")
        if details["expected_actions"] == details["received_actions"]:
            raise ValueError("horizon rejection requires unequal action counts")
    elif kind == "chunk_accepted":
        _non_negative_integer(details["sequence"], "sequence")
        _positive_integer(details["actions"], "actions")
        _finite_number(details["action_age_s"], "action_age_s", minimum=0)
    elif kind in {"action_dequeued", "action_executed", "action_rejected_endpoint"}:
        _non_negative_integer(details["sequence"], "sequence")
        _non_negative_integer(details["action_index"], "action_index")
        _finite_number(details["action_age_s"], "action_age_s", minimum=0)
        if kind == "action_dequeued":
            _non_negative_integer(details["remaining_steps"], "remaining_steps")
    elif kind in {"session_reconnected", "session_reset"}:
        _non_negative_integer(details["generation"], "generation")
    elif kind in {
        "endpoint_observation_failed",
        "endpoint_fallback_failed",
        "endpoint_action_failed",
        "endpoint_close_failed",
    }:
        _error_string(details["error"])
    elif kind == "control_ticks_missed":
        _positive_integer(details["count"], "missed tick count")
    elif kind == "endpoint_task_step":
        _finite_number(details["reward"], "task reward")
        for field in ("success", "terminated", "truncated"):
            if type(details[field]) is not bool:
                raise ValueError(f"{field} must be boolean")


_EVENT_REQUIRED_DETAILS = {
    "observation_ready": ("sequence",),
    "request_dispatched": ("sequence", "batch_size"),
    "chunk_rejected_horizon": ("expected_actions", "received_actions"),
    "chunk_accepted": ("sequence", "actions", "action_age_s"),
    "action_dequeued": (
        "remaining_steps",
        "sequence",
        "action_index",
        "action_age_s",
    ),
    "action_executed": ("sequence", "action_index", "action_age_s"),
    "action_rejected_endpoint": ("sequence", "action_index", "action_age_s"),
    "session_reconnected": ("generation",),
    "session_reset": ("generation",),
    "endpoint_observation_failed": ("error",),
    "endpoint_fallback_failed": ("error",),
    "endpoint_action_failed": ("error",),
    "endpoint_close_failed": ("error",),
    "control_ticks_missed": ("count",),
    "endpoint_task_step": ("reward", "success", "terminated", "truncated"),
}


def _validate_inference_lifecycle(
    events: list[dict[str, Any]],
    *,
    system: bool,
    expected_sessions: set[str],
    duration_s: float,
    max_batch_size: int,
    model_output_horizon: int | None = None,
    execution_horizon: int | None = None,
) -> None:
    stage: str | None = None
    batch_size: int | None = None
    batch_sessions: tuple[str, ...] = ()
    pending_requests: list[tuple[str, int, int]] = []
    batch_sequences: dict[str, int] = {}
    expected_chunks: dict[str, tuple[int, bool, float]] = {}
    expected_resets: set[str] = set()
    invalidated_sessions: set[str] = set()
    registered: list[str] = []
    registration_ended = False
    started_at_s: float | None = None
    modeled_latency_s: float | None = None

    for event in events:
        kind = event["kind"]
        details = event["details"] or {}
        session_id = event["session_id"]
        if not system and event["time_s"] > duration_s:
            raise ValueError("algorithm event occurs after benchmark duration")
        if kind == "session_registered":
            if registration_ended:
                raise ValueError("sessions must be registered at startup")
            registered.append(session_id)
            continue
        registration_ended = True
        if system and kind in {"session_reset", "session_disconnected"}:
            if session_id in batch_sessions:
                invalidated_sessions.add(session_id)
            if session_id in expected_chunks:
                sequence, _, produced_at_s = expected_chunks[session_id]
                expected_chunks[session_id] = (sequence, True, produced_at_s)

        if kind == "request_dispatched":
            if stage is not None or expected_chunks or expected_resets:
                raise ValueError("request dispatched before the prior batch ended")
            pending_requests.append(
                (session_id, details["batch_size"], details["sequence"])
            )
        elif kind == "batch_dispatched":
            selected = tuple(details["session_ids"])
            if (
                stage is not None
                or expected_chunks
                or expected_resets
                or len(selected) > max_batch_size
            ):
                raise ValueError("a batch was dispatched before the prior batch ended")
            expected_requests = [
                (selected_session, details["batch_size"])
                for selected_session in selected
            ]
            if [request[:2] for request in pending_requests] != expected_requests:
                raise ValueError("batch sessions do not match dispatched requests")
            batch_sequences = {
                session: sequence for session, _, sequence in pending_requests
            }
            pending_requests.clear()
            stage = "dispatched"
            batch_size = details["batch_size"]
            batch_sessions = selected
            invalidated_sessions.clear()
        elif kind == "scheduler_cost_estimate":
            if (
                not system
                or stage != "dispatched"
                or details["batch_size"] != batch_size
            ):
                raise ValueError("scheduler cost event does not match its batch")
            stage = "costed"
        elif kind == "inference_started":
            expected_stage = "costed" if system else "dispatched"
            if stage != expected_stage or details["batch_size"] != batch_size:
                raise ValueError("inference start does not match its batch")
            modeled_latency_s = details.get("latency_s")
            if not system and modeled_latency_s is None:
                raise ValueError("algorithm inference start lacks modeled latency")
            started_at_s = event["time_s"]
            stage = "started"
        elif kind == "inference_completed":
            if stage != "started" or details["batch_size"] != batch_size:
                raise ValueError("inference completion does not match its start")
            if system:
                measured_latency_s = details.get("latency_s")
                backend_latency_s = details.get("backend_reported_latency_s")
                shapes = details.get("output_shapes")
                horizons = details.get("execution_horizons")
                if (
                    measured_latency_s is None
                    or backend_latency_s is None
                    or started_at_s is None
                    or not math.isclose(
                        measured_latency_s,
                        event["time_s"] - started_at_s,
                        rel_tol=0,
                        abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                    )
                    or backend_latency_s
                    > measured_latency_s + _EVENT_CLOCK_TOLERANCE_S
                    or not isinstance(shapes, list)
                    or any(
                        len(shape) != 3 or shape[1] != model_output_horizon
                        for shape in shapes
                    )
                    or not isinstance(horizons, list)
                    or any(horizon != execution_horizon for horizon in horizons)
                ):
                    raise ValueError(
                        "inference timing or output does not match the system trace"
                    )
                expected_chunks = {
                    session: (
                        sequence,
                        session in invalidated_sessions,
                        started_at_s + backend_latency_s,
                    )
                    for session, sequence in batch_sequences.items()
                }
            stage = None
            batch_size = None
            batch_sessions = ()
            batch_sequences = {}
            invalidated_sessions.clear()
            started_at_s = None
            modeled_latency_s = None
        elif kind == "inference_failed":
            expected_stage = (
                "costed" if details["phase"] == "prepare" else "started"
            )
            if (
                not system
                or stage != expected_stage
                or tuple(details["session_ids"]) != batch_sessions
            ):
                raise ValueError("inference failure does not match its batch")
            expected_resets = set(batch_sessions)
            stage = None
            batch_size = None
            batch_sessions = ()
            batch_sequences = {}
            invalidated_sessions.clear()
            started_at_s = None
            modeled_latency_s = None
        elif system and kind in {
            "chunk_accepted",
            "chunk_rejected_stale",
            "chunk_rejected_unexpected",
            "chunk_rejected_horizon",
        }:
            if session_id not in expected_chunks:
                raise ValueError("chunk outcome does not match a completed batch")
            expected_sequence, invalidated, produced_at_s = expected_chunks[session_id]
            if invalidated != (kind == "chunk_rejected_stale"):
                raise ValueError("chunk outcome does not match session invalidation")
            if kind == "chunk_accepted" and (
                details["sequence"] != expected_sequence
                or details["actions"] != execution_horizon
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - produced_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("accepted chunk does not match its request")
            if kind == "chunk_rejected_horizon" and (
                details["expected_actions"] != execution_horizon
            ):
                raise ValueError("horizon rejection does not match system config")
            del expected_chunks[session_id]
        elif kind == "session_reset" and session_id in expected_resets:
            expected_resets.remove(session_id)

    if (
        len(registered) != len(expected_sessions)
        or set(registered) != expected_sessions
    ):
        raise ValueError("artifact registrations do not match configured sessions")
    if pending_requests:
        raise ValueError("artifact contains requests without a batch")
    if system and (stage is not None or expected_chunks or expected_resets):
        raise ValueError("system artifact contains an incomplete inference lifecycle")
    if not system and stage not in {None, "started"}:
        raise ValueError("algorithm artifact contains an incomplete dispatch")
    if not system and stage == "started":
        if (
            modeled_latency_s is None
            or started_at_s is None
            or started_at_s + modeled_latency_s <= duration_s
        ):
            raise ValueError("algorithm artifact omits a due inference completion")


def _validate_system_session_lifecycle(
    events: list[dict[str, Any]],
    expected_sessions: set[str],
    *,
    duration_s: float,
    control_hz: float,
) -> None:
    buffers: dict[str, list[tuple[int, int]]] = {
        session_id: [] for session_id in expected_sessions
    }
    outstanding: dict[str, tuple[int, int] | None] = {
        session_id: None for session_id in expected_sessions
    }
    connected = {session_id: True for session_id in expected_sessions}
    generations = {session_id: 0 for session_id in expected_sessions}
    next_sequences = {session_id: 0 for session_id in expected_sessions}
    ready: dict[str, int | None] = {
        session_id: None for session_id in expected_sessions
    }
    observation_times: dict[str, dict[int, float]] = {
        session_id: {} for session_id in expected_sessions
    }
    ready_times: dict[str, float | None] = {
        session_id: None for session_id in expected_sessions
    }
    dispatched_times: dict[str, float | None] = {
        session_id: None for session_id in expected_sessions
    }
    fallback_due = {session_id: False for session_id in expected_sessions}
    task_step_due = {session_id: False for session_id in expected_sessions}
    boundary_due = {session_id: False for session_id in expected_sessions}
    reset_due = {session_id: False for session_id in expected_sessions}
    failure_reset_due = {session_id: False for session_id in expected_sessions}
    action_failure_due = {session_id: False for session_id in expected_sessions}
    disconnect_due = {session_id: False for session_id in expected_sessions}
    ticks = {session_id: 0 for session_id in expected_sessions}

    for event in events:
        kind = event["kind"]
        details = event["details"] or {}
        session_id = event["session_id"]

        if kind == "inference_failed":
            for failed_session in details["session_ids"]:
                failure_reset_due[failed_session] = True
            continue
        if kind == "batch_dispatched":
            for state in details["selected_state"]:
                selected_session = state["session_id"]
                captured_at_s = dispatched_times[selected_session]
                if captured_at_s is None:
                    raise ValueError("selected session has no dispatched observation")
                expected_age_s = event["time_s"] - captured_at_s
                expected_horizon_s = (
                    len(buffers[selected_session]) / control_hz
                )
                if not math.isclose(
                    state["request_age_s"],
                    expected_age_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                ) or not math.isclose(
                    state["buffer_horizon_s"],
                    expected_horizon_s,
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("selected_state does not match session history")
                dispatched_times[selected_session] = None
            continue
        if session_id is None:
            continue
        if dispatched_times[session_id] is not None:
            raise ValueError("dispatched request must enter its batch before local work")
        if outstanding[session_id] is not None and kind not in {
            "action_executed",
            "action_rejected_endpoint",
        }:
            raise ValueError("dequeued action must receive its outcome before local work")
        if failure_reset_due[session_id] and kind != "session_reset":
            raise ValueError("inference failure must be followed by a session reset")
        if fallback_due[session_id] and kind not in {
            "endpoint_fallback",
            "endpoint_fallback_failed",
        }:
            raise ValueError("starved or reset session must complete its fallback")
        if task_step_due[session_id] and kind != "endpoint_task_step":
            raise ValueError("executed action or fallback must report its task step")
        if action_failure_due[session_id] and kind != "endpoint_action_failed":
            raise ValueError("rejected action must report its endpoint failure")
        if disconnect_due[session_id] and kind != "session_disconnected":
            raise ValueError("endpoint failure must be followed by disconnect")
        if boundary_due[session_id] and kind != "endpoint_episode_boundary":
            raise ValueError("terminal task step must be followed by an episode boundary")
        if reset_due[session_id] and kind != "session_reset":
            raise ValueError("episode boundary must be followed by a session reset")

        if kind == "observation_ready":
            if (
                not connected[session_id]
                or ready[session_id] is not None
                or details["sequence"] != next_sequences[session_id]
            ):
                raise ValueError("observation sequence does not match session state")
            ready[session_id] = details["sequence"]
            observation_times[session_id][details["sequence"]] = event["time_s"]
            ready_times[session_id] = event["time_s"]
            next_sequences[session_id] += 1
        elif kind == "request_dispatched":
            if ready[session_id] != details["sequence"]:
                raise ValueError("request does not match a ready observation")
            dispatched_times[session_id] = ready_times[session_id]
            ready[session_id] = None
            ready_times[session_id] = None
        elif kind == "chunk_accepted":
            if not connected[session_id]:
                raise ValueError("a disconnected session accepted a chunk")
            sequence = details["sequence"]
            buffers[session_id].extend(
                (sequence, index) for index in range(details["actions"])
            )
        elif kind == "action_dequeued":
            key = (details["sequence"], details["action_index"])
            captured_at_s = observation_times[session_id].get(details["sequence"])
            if (
                not connected[session_id]
                or outstanding[session_id] is not None
                or not buffers[session_id]
                or buffers[session_id][0] != key
                or captured_at_s is None
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - captured_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("dequeued action is not the next buffered action")
            buffers[session_id].pop(0)
            if details["remaining_steps"] != len(buffers[session_id]):
                raise ValueError("dequeued action reports an invalid remaining count")
            outstanding[session_id] = key
        elif kind in {"action_executed", "action_rejected_endpoint"}:
            key = (details["sequence"], details["action_index"])
            captured_at_s = observation_times[session_id].get(details["sequence"])
            if (
                outstanding[session_id] != key
                or captured_at_s is None
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - captured_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("action outcome does not match a dequeued action")
            outstanding[session_id] = None
            ticks[session_id] += 1
            if kind == "action_executed":
                task_step_due[session_id] = True
            else:
                action_failure_due[session_id] = True
        elif kind == "action_starved":
            if (
                not connected[session_id]
                or buffers[session_id]
                or outstanding[session_id] is not None
            ):
                raise ValueError("session reported starvation with an action available")
            fallback_due[session_id] = True
            ticks[session_id] += 1
        elif kind == "control_ticks_missed":
            ticks[session_id] += details["count"]
        elif kind == "endpoint_fallback":
            if not fallback_due[session_id]:
                raise ValueError("endpoint fallback has no corresponding control tick")
            fallback_due[session_id] = False
            task_step_due[session_id] = True
        elif kind == "endpoint_fallback_failed":
            if not fallback_due[session_id]:
                raise ValueError("failed endpoint fallback was not requested")
            fallback_due[session_id] = False
            disconnect_due[session_id] = True
        elif kind == "endpoint_action_failed":
            if not action_failure_due[session_id]:
                raise ValueError("endpoint action failure has no rejected action")
            action_failure_due[session_id] = False
            disconnect_due[session_id] = True
        elif kind == "endpoint_observation_failed":
            disconnect_due[session_id] = True
        elif kind == "endpoint_task_step":
            if not task_step_due[session_id]:
                raise ValueError("task step has no executed action or fallback")
            task_step_due[session_id] = False
            if details["terminated"] or details["truncated"]:
                boundary_due[session_id] = True
        elif kind == "endpoint_episode_boundary":
            if not boundary_due[session_id]:
                raise ValueError("episode boundary has no terminal task step")
            boundary_due[session_id] = False
            reset_due[session_id] = True
        elif kind == "session_reset":
            after_failure = failure_reset_due[session_id]
            if not (reset_due[session_id] or after_failure):
                raise ValueError("session reset has no failure or episode boundary")
            buffers[session_id].clear()
            outstanding[session_id] = None
            ready[session_id] = None
            ready_times[session_id] = None
            dispatched_times[session_id] = None
            reset_due[session_id] = False
            failure_reset_due[session_id] = False
            generations[session_id] += 1
            if details["generation"] != generations[session_id]:
                raise ValueError("session reset reports an invalid generation")
            if after_failure:
                fallback_due[session_id] = True
        elif kind == "session_disconnected":
            if not connected[session_id] or not disconnect_due[session_id]:
                raise ValueError("session disconnect has no endpoint failure")
            disconnect_due[session_id] = False
            connected[session_id] = False
            buffers[session_id].clear()
            outstanding[session_id] = None
            ready[session_id] = None
            ready_times[session_id] = None
            dispatched_times[session_id] = None
            generations[session_id] += 1
        elif kind == "session_reconnected":
            if connected[session_id]:
                raise ValueError("connected session was reconnected")
            if details["generation"] != generations[session_id]:
                raise ValueError("session reconnect reports an invalid generation")
            connected[session_id] = True

    expected_ticks = math.floor(duration_s * control_hz + 1e-9)
    if any(count != expected_ticks for count in ticks.values()):
        raise ValueError("system event ticks do not match duration and control_hz")
    pending = (
        outstanding,
        fallback_due,
        task_step_due,
        boundary_due,
        reset_due,
        failure_reset_due,
        action_failure_due,
        disconnect_due,
    )
    if any(any(values.values()) for values in pending):
        raise ValueError("system artifact ends with an incomplete session lifecycle")


def _validate_system_artifact(artifact: dict[str, Any]) -> set[str]:
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
    task_ids = config["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(type(task_id) is not int or task_id < 0 for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("system task_ids must be unique non-negative integers")
    expected_sessions = {
        f"{config['suite']}-{task_id}" for task_id in task_ids
    }
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
    if config["action_execution"] != "sequential-buffer":
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


def _validate_benchmark_metrics(metrics: Any, expected_sessions: set[str]) -> None:
    if not isinstance(metrics, dict):
        raise ValueError("benchmark metrics must be an object")
    _require_keys(
        metrics,
        "useful_actions",
        "starvation_frequency",
        "starvation_duration_s",
        "action_age_p50_s",
        "action_age_p95_s",
        "fairness",
        "batch_sizes",
        "backend_utilization",
        "per_session",
    )
    _non_negative_integer(metrics["useful_actions"], "useful_actions")
    _unit_interval(metrics["starvation_frequency"], "starvation_frequency")
    _finite_number(metrics["starvation_duration_s"], "starvation_duration_s", minimum=0)
    for field in ("action_age_p50_s", "action_age_p95_s"):
        if metrics[field] is not None:
            _finite_number(metrics[field], field, minimum=0)
    if (
        metrics["action_age_p50_s"] is not None
        and metrics["action_age_p95_s"] is not None
        and metrics["action_age_p95_s"] < metrics["action_age_p50_s"]
    ):
        raise ValueError("action_age_p95_s cannot be below action_age_p50_s")
    _unit_interval(metrics["fairness"], "fairness")
    _unit_interval(metrics["backend_utilization"], "backend_utilization")
    batches = metrics["batch_sizes"]
    if not isinstance(batches, list) or any(
        type(size) is not int or size <= 0 for size in batches
    ):
        raise ValueError("batch_sizes must contain positive integers")
    per_session = metrics["per_session"]
    if not isinstance(per_session, dict) or set(per_session) != expected_sessions:
        raise ValueError("per_session metrics must match configured sessions")
    for session_metrics in per_session.values():
        if not isinstance(session_metrics, dict):
            raise ValueError("per_session metrics must be objects")
        _require_keys(
            session_metrics,
            "actions",
            "starved_ticks",
            "starvation_duration_s",
            "useful_progress_ratio",
        )
        _non_negative_integer(session_metrics["actions"], "session actions")
        _non_negative_integer(session_metrics["starved_ticks"], "session starved_ticks")
        _finite_number(
            session_metrics["starvation_duration_s"],
            "session starvation_duration_s",
            minimum=0,
        )
        _unit_interval(
            session_metrics["useful_progress_ratio"], "useful_progress_ratio"
        )


def _validate_metrics_match_events(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    robots: tuple[RobotSpec, ...],
    duration_s: float,
) -> None:
    try:
        event_objects = tuple(Event(**event) for event in events)
        calculated = compute_metrics(
            SimulationResult(duration_s, event_objects), robots
        ).as_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact event details are invalid") from error
    if calculated != metrics:
        raise ValueError("artifact metrics do not match its events")


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
            "terminated": sum(
                event["details"]["terminated"] for event in task_events
            ),
            "truncated": sum(
                event["details"]["truncated"] for event in task_events
            ),
        }
        if task_metrics != event_metrics:
            raise ValueError("task metrics do not match system events")


def _require_keys(values: dict[str, Any], *keys: str) -> None:
    missing = set(keys) - set(values)
    if missing:
        raise ValueError(f"artifact is missing fields: {', '.join(sorted(missing))}")


def _event_details(
    event: dict[str, Any],
    *required: str,
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    details = event["details"]
    if not isinstance(details, dict):
        raise ValueError(f"event {event['kind']} must include details")
    _require_keys(details, *required)
    unknown = set(details) - set(required) - set(optional)
    if unknown:
        raise ValueError(
            f"event {event['kind']} has unknown details: "
            f"{', '.join(sorted(unknown))}"
        )
    return details


def _require_exact_keys(values: dict[str, Any], *keys: str) -> None:
    _require_keys(values, *keys)
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError(f"unexpected fields: {', '.join(sorted(unknown))}")


def _positive_integer_value(value: Any, name: str) -> int:
    _positive_integer(value, name)
    return value


def _error_string(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("event error must be a non-empty string")


def _require_strings(values: dict[str, Any], *keys: str) -> None:
    _require_keys(values, *keys)
    for key in keys:
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"{key} must be a non-empty string")


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (minimum is not None and (value <= minimum if strict else value < minimum))
    ):
        qualifier = "positive" if strict else "finite"
        raise ValueError(f"{name} must be {qualifier}")


def _non_negative_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _unit_interval(value: Any, name: str) -> None:
    _finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def trace_config(path: str | Path, scheduler: str) -> BenchmarkConfig:
    artifact = load_artifact(path)
    source = BenchmarkConfig.from_dict(artifact["config"])
    observation_schedule = tuple(
        (float(event["time_s"]), str(event["session_id"]))
        for event in artifact["events"]
        if event["kind"]
        in {"observation_ready", "observation_dropped_backpressure"}
    )
    return replace(
        source,
        scheduler=scheduler,
        scheduler_config={},
        environment="trace",
        trace_source=str(Path(path)),
        observation_schedule=observation_schedule,
    )


def render_timeline(run: BenchmarkRun, *, limit: int = 12) -> str:
    rows: list[str] = []
    for event in run.result.events:
        if event.kind == "batch_dispatched" and event.details:
            sessions = ",".join(event.details["session_ids"])
            reason = event.details.get("reason", "")
            horizons = ",".join(
                f"{state['session_id']}:{state['buffer_horizon_s']:.3f}s"
                for state in event.details.get("selected_state", ())
            )
            rows.append(
                f"{event.time_s:7.3f}s  dispatch  [{sessions}] "
                f"batch={event.details['batch_size']} buffers={{{horizons}}}\n"
                f"           reason: {reason}"
            )
        elif event.kind == "inference_completed" and event.details:
            rows.append(
                f"{event.time_s:7.3f}s  complete  batch={event.details['batch_size']}"
            )
        elif event.kind == "action_starved":
            rows.append(f"{event.time_s:7.3f}s  starved  {event.session_id}")
        if len(rows) >= limit:
            break
    return "\n".join(rows) if rows else "(no dispatch or starvation events)"
