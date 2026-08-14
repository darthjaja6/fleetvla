"""LIBERO vector-environment endpoint adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..endpoints import ExecutionOutcome
from ..types import FieldSpec, SessionConfig


@dataclass(slots=True)
class TaskMetrics:
    reward: float = 0.0
    steps: int = 0
    successes: int = 0
    episodes: int = 0
    terminated: int = 0
    truncated: int = 0


class LiberoVectorAdapter:
    """Expose each LIBERO vector slot as a FleetVLA endpoint.

    The wrapped object follows LIBERO's `BaseVectorEnv.reset(id=...)` and
    `step(action, id=...)` interface. Rewards and task success stay here and
    never enter `SessionSnapshot`.
    """

    def __init__(
        self,
        vector_env: Any,
        session_configs: list[SessionConfig],
        *,
        observation_converter: Callable[[Any], Mapping[str, Any]],
        action_converter: Callable[[Any], Any],
        fallback_action: Callable[[], Any],
        observation_schema: tuple[FieldSpec, ...] = (),
        addressing: str = "indexed",
        reset_kwargs: Mapping[str, Any] | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        if addressing not in {"indexed", "single"}:
            raise ValueError("addressing must be 'indexed' or 'single'")
        if addressing == "single" and len(session_configs) != 1:
            raise ValueError("single addressing requires exactly one session")
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.vector_env = vector_env
        self.addressing = addressing
        self.reset_kwargs = dict(reset_kwargs or {})
        self.max_episode_steps = max_episode_steps
        self.metrics = {config.session_id: TaskMetrics() for config in session_configs}
        self.endpoints = [
            _LiberoEndpoint(
                self,
                index,
                config,
                observation_converter,
                action_converter,
                fallback_action,
                observation_schema,
            )
            for index, config in enumerate(session_configs)
        ]


class _LiberoEndpoint:
    def __init__(
        self,
        adapter: LiberoVectorAdapter,
        index: int,
        session_config: SessionConfig,
        observation_converter: Callable[[Any], Mapping[str, Any]],
        action_converter: Callable[[Any], Any],
        fallback_action: Callable[[], Any],
        observation_schema: tuple[FieldSpec, ...],
    ) -> None:
        self.adapter = adapter
        self.index = index
        self.session_config = session_config
        self.observation_schema = observation_schema
        self._observation_converter = observation_converter
        self._action_converter = action_converter
        self._fallback_action = fallback_action
        self._episode_steps = 0
        self._observation = self._reset()
        self._closed = False

    def _reset(self) -> Any:
        if self.adapter.addressing == "indexed":
            result = self.adapter.vector_env.reset(
                id=[self.index], **self.adapter.reset_kwargs
            )
        else:
            result = self.adapter.vector_env.reset(**self.adapter.reset_kwargs)
        self._episode_steps = 0
        return result[0] if isinstance(result, tuple) else result

    def observe(self) -> Mapping[str, Any]:
        return self._observation_converter(_first(self._observation))

    def execute(self, action: Any) -> ExecutionOutcome:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        converted = [_to_array(self._action_converter(action))]
        if self.adapter.addressing == "indexed":
            result = self.adapter.vector_env.step(converted, id=[self.index])
        else:
            result = self.adapter.vector_env.step(_to_array(converted))
        if len(result) == 4:
            observation, reward, done, info = result
            terminated = done
            truncated = False
            success = bool(_first(done))
        elif len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = _first(terminated) or _first(truncated)
            info_value = _first(info)
            success = bool(
                info_value.get("is_success", info_value.get("success", False))
                if isinstance(info_value, Mapping)
                else False
            )
        else:
            raise ValueError("LIBERO step must return 4 or 5 values")
        self._episode_steps += 1
        if (
            not bool(_first(done))
            and self.adapter.max_episode_steps is not None
            and self._episode_steps >= self.adapter.max_episode_steps
        ):
            done = True
            truncated = True
        self._observation = observation
        metrics = self.adapter.metrics[self.session_config.session_id]
        metrics.reward += float(_first(reward))
        metrics.steps += 1
        metrics.successes += int(success)
        episode_boundary = bool(_first(done))
        if episode_boundary:
            metrics.episodes += 1
            metrics.terminated += int(bool(_first(terminated)))
            metrics.truncated += int(bool(_first(truncated)))
            self._observation = self._reset()
        return ExecutionOutcome(
            episode_boundary=episode_boundary,
            task_reward=float(_first(reward)),
            task_success=success,
            terminated=bool(_first(terminated)),
            truncated=bool(_first(truncated)),
        )

    def fallback(self) -> ExecutionOutcome:
        return self.execute(self._fallback_action())

    def close(self) -> None:
        self._closed = True

    def reconnect(self) -> None:
        if not self._closed:
            raise RuntimeError("endpoint is already connected")
        self._observation = self._reset()
        self._closed = False


def _first(value: Any) -> Any:
    try:
        return value[0]
    except (IndexError, KeyError, TypeError):
        return value


def _to_array(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return value
    return np.asarray(value)
