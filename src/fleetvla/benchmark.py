"""Reproducible benchmark configuration, metrics, artifacts, and replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

from .backend import SyntheticBackend
from .runtime import ACTION_EXECUTION_POLICIES
from .schedulers import Scheduler, create_scheduler
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
    batch_latency_s: tuple[float, ...] = ()
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
        if self.action_execution not in ACTION_EXECUTION_POLICIES:
            raise ValueError(
                f"unsupported action execution policy: {self.action_execution}"
            )
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration must be finite and positive")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        InferenceCostModel(
            self.base_latency_s,
            self.per_item_latency_s,
            self.batch_latency_s,
        )
        if type(self.max_batch_size) is not int or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if self.batch_latency_s and self.max_batch_size > len(self.batch_latency_s):
            raise ValueError("latency profile must cover max_batch_size")
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
                [time_s, session_id] for time_s, session_id in self.observation_schedule
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
            "batch_latency_s",
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
        values["batch_latency_s"] = tuple(values.get("batch_latency_s", ()))
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
        message = (
            f"unsupported benchmark config version {version!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
        raise ValueError(message)
    return BenchmarkConfig.from_dict(data)


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    sent_actions: int
    accepted_actions: int
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
    fleetvla_source_sha256: str
    scheduler_source: str | None
    scheduler_source_sha256: str | None


@contextmanager
def _captured_scheduler(
    specification: str,
    scheduler_config: dict[str, Any],
) -> Iterator[tuple[Scheduler, str | None, str | None]]:
    if ":" not in specification:
        yield create_scheduler(specification, scheduler_config), None, None
        return

    filename, class_name = specification.rsplit(":", 1)
    source_bytes = Path(filename).expanduser().resolve().read_bytes()
    source = source_bytes.decode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="fleetvla-benchmark-") as directory:
        captured_path = Path(directory) / "scheduler.py"
        captured_path.write_bytes(source_bytes)
        scheduler = create_scheduler(f"{captured_path}:{class_name}", scheduler_config)
        yield scheduler, source, source_sha256


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    source_sha256 = fleetvla_source_sha256()
    with _captured_scheduler(config.scheduler, config.scheduler_config) as (
        scheduler,
        scheduler_source,
        scheduler_source_sha256,
    ):
        backend = SyntheticBackend(
            chunk_sizes={robot.session_id: robot.chunk_size for robot in config.robots},
            base_latency_s=config.base_latency_s,
            per_item_latency_s=config.per_item_latency_s,
            batch_latency_s=config.batch_latency_s,
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
            action_execution=config.action_execution,
            observation_schedule=observation_schedule,
        ).run(config.duration_s)
    return BenchmarkRun(
        config,
        result,
        compute_metrics(result, config.robots),
        source_sha256,
        scheduler_source,
        scheduler_source_sha256,
    )


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
        sent = result.count("action_sent_endpoint", robot.session_id)
        accepted = result.count("action_accepted_endpoint", robot.session_id)
        if sent == 0:
            sent = executed + rejected
        if accepted == 0:
            accepted = executed
        missed = sum(
            int(event.details["count"])
            for event in result.events
            if event.kind == "control_ticks_missed"
            and event.session_id == robot.session_id
            and event.details
        )
        starved = result.count("action_starved", robot.session_id) + missed + rejected
        ticks = executed + starved
        starvation_duration_s = starved / robot.control_hz
        ratio = executed / ticks if ticks else 0.0
        progress_ratios.append(ratio)
        total_starved += starved
        total_ticks += ticks
        per_session[robot.session_id] = {
            "sent_actions": sent,
            "accepted_actions": accepted,
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
        sent_actions=sum(
            int(metrics["sent_actions"]) for metrics in per_session.values()
        ),
        accepted_actions=sum(
            int(metrics["accepted_actions"]) for metrics in per_session.values()
        ),
        useful_actions=sum(int(metrics["actions"]) for metrics in per_session.values()),
        starvation_frequency=total_starved / total_ticks if total_ticks else 0.0,
        starvation_duration_s=sum(
            float(metrics["starvation_duration_s"]) for metrics in per_session.values()
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
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "provenance": {
            "fleetvla_version": importlib.metadata.version("fleetvla"),
            "fleetvla_source_sha256": run.fleetvla_source_sha256,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "scheduler_source_sha256": run.scheduler_source_sha256,
            "scheduler_source": run.scheduler_source,
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
    from .artifact import write_artifact as write

    return write(run, path)


def verify_artifact(
    path: str | Path, *, allow_source_mismatch: bool = False
) -> dict[str, Any]:
    from .artifact import verify_artifact as verify

    return verify(path, allow_source_mismatch=allow_source_mismatch)


def load_artifact(path: str | Path) -> dict[str, Any]:
    from .artifact import load_artifact as load

    return load(path)


def replay_artifact(
    path: str | Path, *, allow_embedded_scheduler: bool = False
) -> tuple[BenchmarkRun, bool]:
    from .artifact import replay_artifact as replay

    return replay(path, allow_embedded_scheduler=allow_embedded_scheduler)


def trace_config(path: str | Path, scheduler: str) -> BenchmarkConfig:
    artifact = load_artifact(path)
    source = BenchmarkConfig.from_dict(artifact["config"])
    observation_schedule = tuple(
        (float(event["time_s"]), str(event["session_id"]))
        for event in artifact["events"]
        if event["kind"] in {"observation_ready", "observation_dropped_backpressure"}
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
