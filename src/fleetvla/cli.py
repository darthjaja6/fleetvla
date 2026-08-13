"""FleetVLA command-line interface."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

from .benchmark import (
    BenchmarkConfig,
    default_config,
    load_config,
    render_timeline,
    replay_artifact,
    run_matrix,
    trace_config,
    write_artifact,
    verify_artifact,
)
from .schedulers import check_scheduler, create_scheduler, registry
from .simulation import FleetSimulator, RobotSpec


def _demo(duration_s: float) -> int:
    result = FleetSimulator(
        [
            RobotSpec("fast-arm", control_hz=20, chunk_size=4),
            RobotSpec("slow-arm", control_hz=10, chunk_size=4),
        ],
        max_batch_size=4,
    ).run(duration_s)
    print("FleetVLA deterministic CPU demo")
    print(f"duration: {result.duration_s:.2f}s")
    for session_id in ("fast-arm", "slow-arm"):
        executed = result.count("action_executed", session_id)
        starved = result.count("action_starved", session_id)
        print(f"{session_id}: {executed} actions, {starved} starved ticks")
    batches = [
        event.details["batch_size"]
        for event in result.events
        if event.kind == "batch_dispatched" and event.details
    ]
    print(f"dispatch batch sizes: {batches}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fleetvla",
        description="Serve action-chunking robot policies across a fleet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic CPU demo")
    demo.add_argument("--duration", type=float, default=2.0, metavar="SECONDS")

    test_scheduler = subparsers.add_parser(
        "test-scheduler", help="check a built-in or local scheduler contract"
    )
    test_scheduler.add_argument(
        "scheduler", help="registered name or ./file.py:ClassName"
    )
    test_scheduler.add_argument(
        "--config", default="{}", help="scheduler configuration as JSON"
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="run a reproducible fleet scheduling benchmark"
    )
    benchmark.add_argument("--config", dest="config_path", metavar="FILE")
    benchmark.add_argument(
        "--scheduler",
        action="append",
        help=f"repeat to compare; built-ins: {', '.join(registry.names())}",
    )
    benchmark.add_argument(
        "--scheduler-config", help="configuration as JSON"
    )
    benchmark.add_argument(
        "--environment", choices=("synthetic", "trace")
    )
    benchmark.add_argument("--trace", metavar="ARTIFACT")
    benchmark.add_argument("--duration", type=float)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--output", default="results", metavar="PATH")
    benchmark.add_argument("--timeline", action="store_true")

    replay = subparsers.add_parser(
        "replay", help="rerun an artifact and compare every event and metric"
    )
    replay.add_argument("artifact")
    replay.add_argument(
        "--allow-embedded-scheduler",
        action="store_true",
        help="execute scheduler source embedded in a local-scheduler artifact",
    )

    verify = subparsers.add_parser(
        "verify-artifact", help="verify an algorithm or system artifact"
    )
    verify.add_argument("artifact")
    verify.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="inspect an artifact produced by different FleetVLA source",
    )

    libero = subparsers.add_parser(
        "libero", help="run a measured SmolVLA benchmark on LIBERO"
    )
    libero.add_argument(
        "--model",
        default="HuggingFaceVLA/smolvla_libero",
        help="LeRobot SmolVLA model ID",
    )
    libero.add_argument(
        "--revision",
        default="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        help="model revision recorded in the artifact",
    )
    libero.add_argument("--suite", default="libero_spatial", help="LIBERO suite")
    libero.add_argument(
        "--task", type=int, action="append", dest="tasks", help="repeat for each task session"
    )
    libero.add_argument("--scheduler", default="edf", help="registered name or ./file.py:Class")
    libero.add_argument("--scheduler-config", default="{}", help="scheduler configuration as JSON")
    libero.add_argument("--duration", type=float, default=3.0, help="wall-clock run seconds")
    libero.add_argument("--max-batch-size", type=int, default=2)
    libero.add_argument("--control-hz", type=float, default=20)
    libero.add_argument("--episode-length", type=int, default=100)
    libero.add_argument(
        "--execution-horizon",
        type=int,
        help="actions executed per prediction; default: model n_action_steps",
    )
    libero.add_argument("--seed", type=int, default=0)
    libero.add_argument("--output", default="libero-system-result.json")
    return parser


def _scheduler_config(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("scheduler config must be a JSON object")
    return parsed


def _benchmark(args: argparse.Namespace) -> int:
    scheduler_names = args.scheduler or []
    if args.environment == "trace" or args.trace:
        if not args.trace:
            raise ValueError("--environment trace requires --trace ARTIFACT")
        if not scheduler_names:
            scheduler_names = ["adaptive-slack"]
        base = trace_config(args.trace, scheduler_names[0])
    else:
        base = load_config(args.config_path) if args.config_path else default_config()
        if not scheduler_names:
            scheduler_names = [base.scheduler]
    overrides = {}
    if args.scheduler_config is not None:
        overrides["scheduler_config"] = _scheduler_config(args.scheduler_config)
    if args.environment is not None:
        overrides["environment"] = args.environment
    if args.duration is not None:
        overrides["duration_s"] = args.duration
    if args.seed is not None:
        overrides["seed"] = args.seed
    configs = [replace(base, scheduler=name, **overrides) for name in scheduler_names]
    runs = run_matrix(configs)

    output = Path(args.output)
    if len(runs) == 1 and output.suffix == ".json":
        destinations = [output]
    else:
        destinations = [
            output / f"{run.config.scenario}-{run.config.scheduler}-s{run.config.seed}.json"
            for run in runs
        ]
    print("scheduler       actions  starvation  p95 age  fairness  batches")
    for run, destination in zip(runs, destinations):
        write_artifact(run, destination)
        metrics = run.metrics
        p95 = "n/a" if metrics.action_age_p95_s is None else f"{metrics.action_age_p95_s:.3f}s"
        batch_distribution = dict(sorted(Counter(metrics.batch_sizes).items()))
        print(
            f"{run.config.scheduler:15} {metrics.useful_actions:7d}  "
            f"{metrics.starvation_frequency:10.1%}  {p95:>7}  "
            f"{metrics.fairness:8.3f}  {batch_distribution}"
        )
        print(f"artifact: {destination}")
        if args.timeline:
            print(render_timeline(run))
    return 0


def _libero(args: argparse.Namespace) -> int:
    from .integrations.libero_benchmark import (
        run_smolvla_libero,
        write_system_artifact,
    )

    artifact = run_smolvla_libero(
        model_id=args.model,
        revision=args.revision,
        suite=args.suite,
        task_ids=tuple(args.tasks or (0, 1)),
        scheduler_name=args.scheduler,
        scheduler_config=_scheduler_config(args.scheduler_config),
        duration_s=args.duration,
        max_batch_size=args.max_batch_size,
        control_hz=args.control_hz,
        episode_length=args.episode_length,
        execution_horizon=args.execution_horizon,
        seed=args.seed,
    )
    destination = write_system_artifact(artifact, args.output)
    system = artifact["metrics"]["system"]
    failures = [
        event for event in artifact["events"] if "failed" in event["kind"]
    ]
    print(
        f"actions: {system['useful_actions']}, "
        f"starvation: {system['starvation_frequency']:.1%}, "
        f"batches: {system['batch_sizes']}"
    )
    print(f"artifact: {destination}")
    if failures:
        print(f"error: {len(failures)} serving failure events", file=sys.stderr)
        return 1
    return 0


def _run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _demo(args.duration)
    if args.command == "test-scheduler":
        config = _scheduler_config(args.config)
        passed = check_scheduler(lambda: create_scheduler(args.scheduler, config))
        print(f"scheduler {args.scheduler!r} passed {len(passed)} checks")
        for check in passed:
            print(f"  ok  {check}")
        return 0
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "libero":
        return _libero(args)
    if args.command == "replay":
        _, matches = replay_artifact(
            args.artifact,
            allow_embedded_scheduler=args.allow_embedded_scheduler,
        )
        if not matches:
            print("replay mismatch: events or metrics differ")
            return 1
        print("replay matched every event and metric")
        return 0
    if args.command == "verify-artifact":
        artifact = verify_artifact(
            args.artifact,
            allow_source_mismatch=args.allow_source_mismatch,
        )
        scope = "historical-source " if args.allow_source_mismatch else ""
        print(
            f"verified {scope}{artifact.get('artifact_kind', 'algorithm')} artifact: "
            f"version {artifact['artifact_version']}, sha256 {artifact['sha256']}"
        )
        return 0
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
