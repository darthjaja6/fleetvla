"""End-to-end SmolVLA system benchmark on independent LIBERO environments."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from ..benchmark import _captured_scheduler, fleetvla_source_sha256
from ..runtime import ACTION_EXECUTION_POLICIES
from ..serving import AsyncServingEngine, serving_metrics
from ..trace import Event
from ..types import SessionConfig
from .lerobot import LeRobotPolicyBackend
from .libero import LiberoVectorAdapter

SYSTEM_ARTIFACT_VERSION = 1


def run_smolvla_libero(
    *,
    model_id: str,
    revision: str | None,
    suite: str,
    task_ids: tuple[int, ...],
    scheduler_name: str,
    scheduler_config: dict[str, Any],
    duration_s: float,
    max_batch_size: int,
    control_hz: float = 20,
    episode_length: int = 100,
    execution_horizon: int | None = None,
    action_execution: str = "sequential-buffer",
    seed: int = 0,
    scheduler_timeout_s: float = 0.01,
) -> dict[str, Any]:
    """Run a measured wall-clock benchmark and return a JSON-ready artifact."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be non-empty and unique")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    if max_batch_size <= 0 or episode_length <= 0:
        raise ValueError("batch size and episode length must be positive")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    if execution_horizon is not None and execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    if action_execution not in ACTION_EXECUTION_POLICIES:
        raise ValueError("unsupported action execution policy")
    if not math.isfinite(scheduler_timeout_s) or scheduler_timeout_s <= 0:
        raise ValueError("scheduler timeout must be finite and positive")

    source_sha256 = fleetvla_source_sha256()
    with _captured_scheduler(scheduler_name, scheduler_config) as captured:
        scheduler, scheduler_source, scheduler_source_sha256 = captured

    try:
        import numpy as np
        import torch
        from lerobot.envs import (
            make_env,
            make_env_pre_post_processors,
            preprocess_observation,
        )
        from lerobot.envs.configs import LiberoEnv
    except ImportError as error:
        raise ImportError(
            "LIBERO system benchmarks require Python 3.12+ and the "
            "'smolvla' and 'libero' extras"
        ) from error

    backend = LeRobotPolicyBackend.from_smolvla_pretrained(
        model_id,
        revision=revision,
        execution_horizon=execution_horizon,
        max_batch_size=max_batch_size,
    )
    adapters: list[LiberoVectorAdapter] = []
    environments = []
    descriptions: dict[str, str] = {}
    try:
        for task_id in task_ids:
            env_config = LiberoEnv(
                task=suite,
                task_ids=[task_id],
                episode_length=episode_length,
                observation_height=256,
                observation_width=256,
            )
            environment = make_env(env_config, n_envs=1)[suite][task_id]
            environments.append(environment)
            env_preprocessor, _ = make_env_pre_post_processors(
                env_config, backend.policy.config
            )
            task = list(environment.call("task_description"))
            session_id = f"{suite}-{task_id}"
            descriptions[session_id] = task[0]

            def convert(
                observation: Any,
                env_preprocessor: Any = env_preprocessor,
                task: list[str] = task,
            ) -> dict[str, Any]:
                payload = preprocess_observation(observation)
                payload["task"] = task
                return cast(dict[str, Any], env_preprocessor(payload))

            action_dim = int(environment.single_action_space.shape[-1])

            def fallback_action(action_dim: int = action_dim) -> Any:
                return np.zeros(action_dim, dtype=np.float32)

            adapter = LiberoVectorAdapter(
                environment,
                [
                    SessionConfig(
                        session_id,
                        control_hz=control_hz,
                        chunk_size=(
                            backend.execution_horizon
                            or int(backend.policy.config.chunk_size)
                        ),
                    )
                ],
                observation_converter=convert,
                action_converter=lambda action: action,
                fallback_action=fallback_action,
                addressing="single",
                reset_kwargs={"seed": seed + task_id},
                max_episode_steps=episode_length,
            )
            adapters.append(adapter)
    except Exception:
        _close_environments(environments)
        raise

    endpoints = [adapter.endpoints[0] for adapter in adapters]
    inference_timeout_s = max(10.0, duration_s)
    endpoint_timeout_s = 0.1
    try:
        engine = AsyncServingEngine(
            endpoints,
            backend,
            scheduler,
            max_batch_size=max_batch_size,
            inference_timeout_s=inference_timeout_s,
            scheduler_timeout_s=scheduler_timeout_s,
            endpoint_timeout_s=endpoint_timeout_s,
            action_execution=action_execution,
        )
    except Exception:
        _close_environments(environments)
        raise

    async def run_and_close() -> tuple[Event, ...]:
        try:
            return await engine.run(duration_s)
        finally:
            await engine.aclose()

    try:
        events = asyncio.run(run_and_close())
        metrics = serving_metrics(events, endpoints, duration_s)
    finally:
        _close_environments(environments)

    body = {
        "artifact_version": SYSTEM_ARTIFACT_VERSION,
        "artifact_kind": "system",
        "provenance": {
            "fleetvla_version": importlib.metadata.version("fleetvla"),
            "fleetvla_source_sha256": source_sha256,
            "lerobot_version": importlib.metadata.version("lerobot"),
            "libero_version": importlib.metadata.version("hf-libero"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "accelerator": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "scheduler_source": scheduler_source,
            "scheduler_source_sha256": scheduler_source_sha256,
        },
        "config": {
            "clock": "monotonic-wall",
            "environment": "libero",
            "suite": suite,
            "task_ids": list(task_ids),
            "task_descriptions": descriptions,
            "model_id": model_id,
            "model_revision": revision,
            "scheduler": scheduler_name,
            "scheduler_config": scheduler_config,
            "scheduler_timeout_s": scheduler_timeout_s,
            "inference_timeout_s": inference_timeout_s,
            "endpoint_timeout_s": endpoint_timeout_s,
            "duration_s": duration_s,
            "max_batch_size": max_batch_size,
            "control_hz": control_hz,
            "model_output_horizon": int(backend.policy.config.chunk_size),
            "execution_horizon": (
                backend.execution_horizon or int(backend.policy.config.chunk_size)
            ),
            "action_execution": action_execution,
            "episode_length": episode_length,
            "seed": seed,
        },
        "metrics": {
            "system": metrics.as_dict(),
            "tasks": {
                session_id: asdict(task_metrics)
                for adapter in adapters
                for session_id, task_metrics in adapter.metrics.items()
            },
        },
        "events": [event.as_dict() for event in events],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "sha256": digest}


def write_system_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _close_environments(environments: list[Any]) -> None:
    for environment in environments:
        try:
            environment.close()
        except Exception:
            pass
