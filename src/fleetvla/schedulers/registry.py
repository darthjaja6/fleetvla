"""Explicit built-in registry and direct local scheduler loading."""

from __future__ import annotations

import importlib.util
import types
from dataclasses import fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .base import Scheduler


class SchedulerRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type] = {}

    def register(self, name: str, scheduler_class: type) -> None:
        if not name or name in self._classes:
            raise ValueError(f"invalid or duplicate scheduler name: {name!r}")
        self._classes[name] = scheduler_class

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

    def create(
        self, specification: str, config: dict[str, Any] | None = None
    ) -> Scheduler:
        config = config or {}
        scheduler_class = (
            self._load_local(specification)
            if ":" in specification
            else self._classes.get(specification)
        )
        if scheduler_class is None:
            available = ", ".join(self.names())
            raise ValueError(
                f"unknown scheduler {specification!r}; available: {available}"
            )
        if not hasattr(scheduler_class, "schedule"):
            raise TypeError(f"scheduler has no schedule method: {specification}")
        config_type = getattr(scheduler_class, "config_type", None)
        if config_type is None:
            return scheduler_class(**config)
        known = {field.name for field in fields(config_type)}
        unknown = set(config) - known
        if unknown:
            raise ValueError(
                f"unknown config for {specification}: {', '.join(sorted(unknown))}"
            )
        hints = get_type_hints(config_type)
        validated = {
            name: _validate_value(name, value, hints[name])
            for name, value in config.items()
        }
        return scheduler_class(config_type(**validated))

    @staticmethod
    def _load_local(specification: str) -> type:
        filename, class_name = specification.rsplit(":", 1)
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"scheduler file does not exist: {path}")
        module_spec = importlib.util.spec_from_file_location(
            f"fleetvla_local_scheduler_{path.stem}", path
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load scheduler module: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        try:
            return getattr(module, class_name)
        except AttributeError as error:
            raise ValueError(
                f"scheduler class {class_name!r} not found in {path}"
            ) from error


def _validate_value(name: str, value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        options = get_args(annotation)
        if value is None and type(None) in options:
            return None
        non_none = tuple(option for option in options if option is not type(None))
        if len(non_none) == 1:
            return _validate_value(name, value, non_none[0])
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        expected = "an integer"
    elif annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        expected = "a number"
    elif annotation is bool:
        if isinstance(value, bool):
            return value
        expected = "a boolean"
    elif annotation is str:
        if isinstance(value, str):
            return value
        expected = "a string"
    else:
        raise TypeError(
            f"unsupported scheduler config annotation for {name}: {annotation}"
        )
    actual = type(value).__name__
    raise ValueError(f"{name} must be {expected}, got {actual}")
