"""Endpoint contract and local safety wrappers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol, cast

from .types import FieldSpec, SessionConfig


class ObservationUnavailable(RuntimeError):
    """A connected asynchronous endpoint has not produced a new sample yet."""


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    episode_boundary: bool = False
    task_reward: float | None = None
    task_success: bool | None = None
    terminated: bool | None = None
    truncated: bool | None = None

    def __post_init__(self) -> None:
        task_values = (
            self.task_reward,
            self.task_success,
            self.terminated,
            self.truncated,
        )
        if any(value is not None for value in task_values):
            if any(value is None for value in task_values):
                raise ValueError("task outcome fields must be provided together")
            if not isinstance(self.task_reward, (int, float)) or not math.isfinite(
                self.task_reward
            ):
                raise ValueError("task reward must be finite")
            if any(
                type(value) is not bool
                for value in (self.task_success, self.terminated, self.truncated)
            ):
                raise ValueError("task outcome flags must be boolean")
            if self.episode_boundary != (self.terminated or self.truncated):
                raise ValueError("episode boundary must match task termination")


class Endpoint(Protocol):
    session_config: SessionConfig
    observation_schema: tuple[FieldSpec, ...]

    def observe(self) -> Mapping[str, Any]: ...

    def execute(self, action: Any) -> ExecutionOutcome | None: ...

    def fallback(self) -> ExecutionOutcome | None: ...

    def close(self) -> None: ...

    def reconnect(self) -> None: ...


def validate_observation(
    payload: Mapping[str, Any], schema: tuple[FieldSpec, ...]
) -> None:
    for field in schema:
        field.validate(payload)


class LeRobotRobotEndpoint:
    """Safety boundary for a connected LeRobot-compatible physical robot.

    FleetVLA never imports LeRobot here. The wrapped object follows LeRobot's
    current `get_observation`, `send_action`, and `disconnect` interface.
    """

    def __init__(
        self,
        robot: Any,
        session_config: SessionConfig,
        *,
        observation_schema: tuple[FieldSpec, ...] = (),
        action_converter: Callable[[Any], Mapping[str, Any]],
        safe_action: Callable[[], Mapping[str, Any]],
        action_validator: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.robot = robot
        self.session_config = session_config
        self.observation_schema = observation_schema
        self._action_converter = action_converter
        self._safe_action = safe_action
        self._action_validator = action_validator or _finite_mapping
        self._closed = False

    def observe(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        observation = cast(Mapping[str, Any], self.robot.get_observation())
        validate_observation(observation, self.observation_schema)
        return observation

    def execute(self, action: Any) -> None:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        converted = self._action_converter(action)
        if not self._action_validator(converted):
            self.fallback()
            raise ValueError("action rejected by the endpoint safety validator")
        self.robot.send_action(converted)

    def fallback(self) -> None:
        if not self._closed:
            safe_action = self._safe_action()
            if not self._action_validator(safe_action):
                raise ValueError("safe fallback failed the endpoint validator")
            self.robot.send_action(safe_action)

    def close(self) -> None:
        if not self._closed:
            try:
                self.fallback()
            finally:
                try:
                    self.robot.disconnect()
                finally:
                    self._closed = True

    def reconnect(self) -> None:
        if not self._closed:
            raise RuntimeError("endpoint is already connected")
        self.robot.connect()
        self._closed = False


def _finite_mapping(action: Mapping[str, Any]) -> bool:
    return all(_finite_value(value) for value in action.values())


def _finite_value(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_value(item) for item in value.values())
    if isinstance(value, (str, bytes, bytearray)):
        return False
    fields = getattr(value, "get_fields_and_field_types", None)
    if callable(fields):
        return all(_finite_value(getattr(value, name)) for name in fields())
    slots = getattr(type(value), "__slots__", ())
    if slots:
        names = (slots,) if isinstance(slots, str) else slots
        try:
            return all(
                _finite_value(getattr(value, name.lstrip("_"), getattr(value, name)))
                for name in names
            )
        except AttributeError:
            return False
    flat = getattr(value, "reshape", lambda *_: value)(-1)
    try:
        return all(math.isfinite(float(item)) for item in flat)
    except TypeError:
        try:
            return all(_finite_value(item) for item in value)
        except TypeError:
            return False


class ROS2Endpoint:
    """ROS 2 topic bridge with conversion and fallback kept at the endpoint."""

    def __init__(
        self,
        node: Any,
        session_config: SessionConfig,
        *,
        observation_topic: str,
        observation_message_type: type,
        observation_converter: Callable[[Any], Mapping[str, Any]],
        action_topic: str,
        action_message_type: type,
        action_converter: Callable[[Any], Any],
        fallback_message: Callable[[], Any],
        observation_schema: tuple[FieldSpec, ...] = (),
        action_validator: Callable[[Any], bool] | None = None,
    ) -> None:
        self.session_config = session_config
        self.observation_schema = observation_schema
        self._observation_converter = observation_converter
        self._action_converter = action_converter
        self._fallback_message = fallback_message
        self._action_validator = action_validator or _finite_value
        self._latest_observation: Any | None = None
        self._observation_lock = Lock()
        self._closed = False
        self._publisher = node.create_publisher(action_message_type, action_topic, 10)
        self._subscription = node.create_subscription(
            observation_message_type,
            observation_topic,
            self._on_observation,
            10,
        )

    def _on_observation(self, message: Any) -> None:
        with self._observation_lock:
            if not self._closed:
                self._latest_observation = message

    def observe(self) -> Mapping[str, Any]:
        with self._observation_lock:
            if self._closed:
                raise RuntimeError("endpoint is closed")
            message = self._latest_observation
            self._latest_observation = None
        if message is None:
            raise ObservationUnavailable("no ROS 2 observation has been received")
        observation = self._observation_converter(message)
        validate_observation(observation, self.observation_schema)
        return observation

    def execute(self, action: Any) -> None:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        message = self._action_converter(action)
        if not self._action_validator(message):
            self.fallback()
            raise ValueError("action rejected by the endpoint safety validator")
        self._publisher.publish(message)

    def fallback(self) -> None:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        message = self._fallback_message()
        if not self._action_validator(message):
            raise ValueError("safe fallback failed the endpoint validator")
        self._publisher.publish(message)

    def close(self) -> None:
        if not self._closed:
            try:
                self.fallback()
            finally:
                with self._observation_lock:
                    self._closed = True
                    self._latest_observation = None

    def reconnect(self) -> None:
        with self._observation_lock:
            if not self._closed:
                raise RuntimeError("endpoint is already connected")
            self._latest_observation = None
            self._closed = False
