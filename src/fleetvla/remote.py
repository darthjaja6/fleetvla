"""Versioned JSON-lines transport for remote robot endpoints."""

from __future__ import annotations

import json
import math
import socket
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .endpoints import ObservationUnavailable, validate_observation
from .types import FieldSpec, SessionConfig

REMOTE_PROTOCOL_VERSION = 1


class RemoteTransport(Protocol):
    """Observation/action channel admitted for one configured session."""

    session_id: str
    protocol_version: int

    def receive_observation(self) -> Mapping[str, Any]: ...

    def send_action(self, action: Any) -> None: ...

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
    ) -> None:
        if transport.protocol_version != REMOTE_PROTOCOL_VERSION:
            raise ValueError("remote transport protocol version is unsupported")
        if transport.session_id != session_config.session_id:
            raise ValueError("remote transport session was not admitted")
        self.transport = transport
        self.session_config = session_config
        self.observation_schema = observation_schema
        self._action_converter = action_converter
        self._safe_action = safe_action
        self._action_validator = action_validator or _finite_value
        self._closed = False

    def observe(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        observation = self.transport.receive_observation()
        validate_observation(observation, self.observation_schema)
        return observation

    def execute(self, action: Any) -> None:
        if self._closed:
            raise RuntimeError("endpoint is closed")
        converted = self._action_converter(action)
        if not self._action_validator(converted):
            self.fallback()
            raise ValueError("action rejected by the remote endpoint validator")
        self.transport.send_action(converted)

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
        self._latest: tuple[int, Mapping[str, Any]] | None = None
        self._last_sequence = -1
        self._error: str | None = None
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

    def send_action(self, action: Any) -> None:
        self._send("action", action)

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
                if (
                    message.get("type") != "observation"
                    or message.get("protocol_version") != REMOTE_PROTOCOL_VERSION
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
                        raise ValueError(
                            "remote observation sequence is not increasing"
                        )
                    self._last_sequence = message["sequence"]
                    self._latest = (message["sequence"], message["payload"])
            with self._state_lock:
                if not self._closed:
                    self._error = "remote peer closed"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            with self._state_lock:
                if not self._closed:
                    self._error = str(error)

    def _send(self, message_type: str, payload: Any) -> None:
        if self._closed:
            raise RuntimeError("remote transport is closed")
        message = {
            "type": message_type,
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "session_id": self.session_id,
        }
        if payload is not None:
            message["payload"] = payload
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._write_lock:
            self._connection.sendall(encoded)

    def _close_connection(self) -> None:
        self._closed = True
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
