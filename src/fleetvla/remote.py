"""Versioned JSON-lines transport for remote robot endpoints."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .endpoints import ObservationUnavailable, validate_observation
from .types import ActionCommand, FieldSpec, SessionConfig

REMOTE_PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class RemoteActionReceipt:
    """Robot-confirmed terminal state for one versioned action command."""

    accepted: bool
    executed: bool

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or type(self.executed) is not bool:
            raise TypeError("remote action receipt fields must be boolean")
        if self.executed and not self.accepted:
            raise ValueError("an executed remote action must first be accepted")


class RemoteActionFailure(RuntimeError):
    """Terminal acknowledgement failed after the robot accepted a command."""

    accepted = True


class RemoteTransport(Protocol):
    """Observation/action channel admitted for one configured session."""

    session_id: str
    protocol_version: int

    def receive_observation(self) -> Mapping[str, Any]: ...

    def send_action(
        self, command: ActionCommand, action: Any, timeout_s: float
    ) -> RemoteActionReceipt: ...

    def send_fallback(self, action: Any) -> None: ...

    def close(self) -> None: ...

    def reconnect(self) -> None: ...


class RemoteEndpoint:
    """Endpoint adapter that keeps remote admission and fallback local."""

    def __init__(
        self,
        transport: RemoteTransport,
        session_config: SessionConfig,
        *,
        observation_schema: tuple[FieldSpec, ...] = (),
        action_converter: Callable[[Any], Any] = lambda action: action,
        safe_action: Callable[[], Any],
        action_validator: Callable[[Any], bool] | None = None,
        acknowledgement_timeout_s: float = 0.1,
    ) -> None:
        if transport.protocol_version != REMOTE_PROTOCOL_VERSION:
            raise ValueError("remote transport protocol version is unsupported")
        if transport.session_id != session_config.session_id:
            raise ValueError("remote transport session was not admitted")
        if (
            not math.isfinite(acknowledgement_timeout_s)
            or acknowledgement_timeout_s <= 0
        ):
            raise ValueError("acknowledgement_timeout_s must be positive")
        self.transport = transport
        self.session_config = session_config
        self.observation_schema = observation_schema
        self._action_converter = action_converter
        self._safe_action = safe_action
        self._action_validator = action_validator or _finite_value
        self._acknowledgement_timeout_s = acknowledgement_timeout_s
        self._closed = False

    def observe(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        observation = self.transport.receive_observation()
        validate_observation(observation, self.observation_schema)
        return observation

    def execute(self, action: Any) -> None:
        del action
        raise RuntimeError(
            "remote actions require command-aware execution by AsyncServingEngine"
        )

    async def execute_command(self, command: ActionCommand) -> RemoteActionReceipt:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        if command.session_id != self.session_config.session_id:
            raise ValueError("action command belongs to another remote session")
        if not math.isfinite(command.deadline_s):
            raise ValueError("remote action command requires a finite deadline")
        converted = self._action_converter(command.value)
        if not self._action_validator(converted):
            raise ValueError("action rejected by the remote endpoint validator")
        return await asyncio.to_thread(
            self.transport.send_action,
            command,
            converted,
            self._acknowledgement_timeout_s,
        )

    def fallback(self) -> None:
        if self._closed:
            return
        action = self._safe_action()
        if not self._action_validator(action):
            raise ValueError("safe fallback failed the remote endpoint validator")
        self.transport.send_fallback(action)

    def close(self) -> None:
        if not self._closed:
            try:
                self.fallback()
            finally:
                self.transport.close()
                self._closed = True

    def reconnect(self) -> None:
        if not self._closed:
            raise RuntimeError("endpoint is already connected")
        self.transport.reconnect()
        if (
            self.transport.protocol_version != REMOTE_PROTOCOL_VERSION
            or self.transport.session_id != self.session_config.session_id
        ):
            raise ValueError("reconnected remote transport was not admitted")
        self._closed = False


class JsonlSocketTransport:
    """One-session TCP client using the public FleetVLA JSON-lines protocol."""

    protocol_version = REMOTE_PROTOCOL_VERSION

    def __init__(
        self,
        connection: socket.socket,
        session_id: str,
        *,
        admission_timeout_s: float = 2.0,
        reconnect_address: tuple[str, int] | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("remote session_id must not be empty")
        if not math.isfinite(admission_timeout_s) or admission_timeout_s <= 0:
            raise ValueError("admission_timeout_s must be positive")
        self.session_id = session_id
        self._admission_timeout_s = admission_timeout_s
        self._reconnect_address = reconnect_address
        self._connection = connection
        self._reader: Any = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        self._latest: tuple[int, Mapping[str, Any]] | None = None
        self._last_sequence = -1
        self._error: str | None = None
        self._pending_actions: dict[tuple[int, int, int], tuple[str, bool]] = {}
        self._closed = False
        self._reader_thread: threading.Thread | None = None
        try:
            self._admit_and_start()
        except BaseException:
            self._close_connection()
            raise

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        session_id: str,
        *,
        admission_timeout_s: float = 2.0,
    ) -> "JsonlSocketTransport":
        address = (host, port)
        connection = socket.create_connection(address, timeout=admission_timeout_s)
        return cls(
            connection,
            session_id,
            admission_timeout_s=admission_timeout_s,
            reconnect_address=address,
        )

    def receive_observation(self) -> Mapping[str, Any]:
        with self._state_lock:
            if self._error is not None:
                raise RuntimeError(f"remote transport failed: {self._error}")
            if self._latest is None:
                raise ObservationUnavailable("no remote observation is available")
            _, payload = self._latest
            self._latest = None
            return payload

    def send_action(
        self, command: ActionCommand, action: Any, timeout_s: float
    ) -> RemoteActionReceipt:
        if command.session_id != self.session_id:
            raise ValueError("action command belongs to another remote session")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("action acknowledgement timeout must be positive")
        key = (
            command.generation,
            command.observation_sequence,
            command.action_index,
        )
        with self._state_changed:
            self._raise_if_unusable()
            if key in self._pending_actions:
                raise RuntimeError("remote action command is already pending")
            self._pending_actions[key] = ("sent", False)
        try:
            self._send_action(command, action)
            expires_at = time.monotonic() + timeout_s
            with self._state_changed:
                while True:
                    self._raise_if_unusable()
                    status, accepted = self._pending_actions[key]
                    if status == "executed":
                        return RemoteActionReceipt(accepted=True, executed=True)
                    if status == "rejected":
                        return RemoteActionReceipt(accepted=accepted, executed=False)
                    remaining_s = expires_at - time.monotonic()
                    if remaining_s <= 0:
                        raise TimeoutError("remote action acknowledgement timed out")
                    self._state_changed.wait(remaining_s)
        except Exception as error:
            with self._state_changed:
                pending = self._pending_actions.get(key)
                accepted = pending is not None and pending[1]
            if accepted:
                raise RemoteActionFailure(
                    "remote action was accepted but terminal acknowledgement failed"
                ) from error
            raise
        finally:
            with self._state_changed:
                self._pending_actions.pop(key, None)

    def send_fallback(self, action: Any) -> None:
        self._send("fallback", action)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._send("close", None)
        except (OSError, RuntimeError):
            pass
        self._close_connection()

    def reconnect(self) -> None:
        if not self._closed:
            raise RuntimeError("remote transport is already connected")
        if self._reconnect_address is None:
            raise RuntimeError("this socket transport cannot reconnect")
        self._connection = socket.create_connection(
            self._reconnect_address, timeout=self._admission_timeout_s
        )
        with self._state_lock:
            self._latest = None
            self._last_sequence = -1
            self._error = None
            self._pending_actions.clear()
        self._closed = False
        try:
            self._admit_and_start()
        except BaseException:
            self._close_connection()
            raise

    def _admit_and_start(self) -> None:
        self._connection.settimeout(self._admission_timeout_s)
        self._reader = self._connection.makefile("r", encoding="utf-8")
        line = self._reader.readline()
        if not line:
            self._close_connection()
            raise ValueError("remote peer closed before admission")
        message = _decode_message(line)
        if (
            message.get("type") != "hello"
            or message.get("protocol_version") != REMOTE_PROTOCOL_VERSION
            or message.get("session_id") != self.session_id
            or set(message) != {"type", "protocol_version", "session_id"}
        ):
            self._close_connection()
            raise ValueError("remote peer admission handshake is invalid")
        self._connection.settimeout(None)
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name=f"fleetvla-remote-{self.session_id}",
            daemon=True,
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        try:
            for line in self._reader:
                message = _decode_message(line)
                if message.get("type") == "observation":
                    self._receive_observation(message)
                elif message.get("type") == "action_ack":
                    self._receive_action_ack(message)
                else:
                    raise ValueError("unsupported remote message type")
            with self._state_changed:
                if not self._closed:
                    self._error = "remote peer closed"
                    self._state_changed.notify_all()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            with self._state_changed:
                if not self._closed:
                    self._error = str(error)
                    self._state_changed.notify_all()

    def _receive_observation(self, message: dict[str, Any]) -> None:
        if (
            message.get("protocol_version") != REMOTE_PROTOCOL_VERSION
            or message.get("session_id") != self.session_id
            or set(message)
            != {
                "type",
                "protocol_version",
                "session_id",
                "sequence",
                "payload",
            }
            or type(message.get("sequence")) is not int
            or message["sequence"] < 0
            or not isinstance(message.get("payload"), Mapping)
        ):
            raise ValueError("invalid remote observation envelope")
        with self._state_lock:
            if message["sequence"] <= self._last_sequence:
                raise ValueError("remote observation sequence is not increasing")
            self._last_sequence = message["sequence"]
            self._latest = (message["sequence"], message["payload"])

    def _receive_action_ack(self, message: dict[str, Any]) -> None:
        fields = {
            "type",
            "protocol_version",
            "session_id",
            "generation",
            "observation_sequence",
            "action_index",
            "status",
        }
        integers = (
            message.get("generation"),
            message.get("observation_sequence"),
            message.get("action_index"),
        )
        if (
            message.get("protocol_version") != REMOTE_PROTOCOL_VERSION
            or message.get("session_id") != self.session_id
            or set(message) != fields
            or any(type(value) is not int or value < 0 for value in integers)
            or message.get("status") not in {"accepted", "executed", "rejected"}
        ):
            raise ValueError("invalid remote action acknowledgement")
        key = integers
        with self._state_changed:
            previous = self._pending_actions.get(key)
            status = message["status"]
            allowed = {
                "sent": {"accepted", "rejected"},
                "accepted": {"executed", "rejected"},
            }
            if previous is None or status not in allowed.get(previous[0], set()):
                raise ValueError("unexpected remote action acknowledgement")
            self._pending_actions[key] = (
                status,
                previous[1] or status in {"accepted", "executed"},
            )
            self._state_changed.notify_all()

    def _send_action(self, command: ActionCommand, payload: Any) -> None:
        deadline_after_s = command.deadline_s - time.monotonic()
        if deadline_after_s <= 0:
            raise TimeoutError("remote action command deadline has passed")
        self._send(
            "action",
            payload,
            generation=command.generation,
            observation_sequence=command.observation_sequence,
            action_index=command.action_index,
            deadline_after_s=deadline_after_s,
        )

    def _send(self, message_type: str, payload: Any, **fields: Any) -> None:
        if self._closed:
            raise RuntimeError("remote transport is closed")
        message = {
            "type": message_type,
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "session_id": self.session_id,
            **fields,
        }
        if payload is not None:
            message["payload"] = payload
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._write_lock:
            self._connection.sendall(encoded)

    def _raise_if_unusable(self) -> None:
        if self._closed:
            raise RuntimeError("remote transport is closed")
        if self._error is not None:
            raise RuntimeError(f"remote transport failed: {self._error}")

    def _close_connection(self) -> None:
        self._closed = True
        with self._state_changed:
            self._state_changed.notify_all()
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._reader.close()
        except (AttributeError, OSError):
            pass
        self._connection.close()


def _decode_message(line: str) -> dict[str, Any]:
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("remote protocol messages must be JSON objects")
    return message


def _finite_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _finite_value(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_finite_value(item) for item in value)
    return value is None or isinstance(value, str)
