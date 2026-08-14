"""Run one acknowledged remote robot session over localhost."""

from __future__ import annotations

import asyncio
import json
import socket
import threading

from fleetvla import (
    FIFOScheduler,
    JsonlSocketTransport,
    RemoteEndpoint,
    SessionConfig,
    SyntheticBackend,
)
from fleetvla.serving import AsyncServingEngine
from fleetvla.trace import Event

SESSION_ID = "arm"
PROTOCOL_VERSION = 2


def _send(connection: socket.socket, message: dict[str, object]) -> None:
    connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())


def _robot(listener: socket.socket, fallback_actions: list[object]) -> None:
    connection, _ = listener.accept()
    reader = connection.makefile("r", encoding="utf-8")
    try:
        _send(
            connection,
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "session_id": SESSION_ID,
            },
        )
        sequence = 0
        _send_observation(connection, sequence)
        for line in reader:
            message = json.loads(line)
            if message["type"] == "action":
                identity = {
                    key: message[key]
                    for key in (
                        "generation",
                        "observation_sequence",
                        "action_index",
                    )
                }
                for status in ("accepted", "executed"):
                    _send(
                        connection,
                        {
                            "type": "action_ack",
                            "protocol_version": PROTOCOL_VERSION,
                            "session_id": SESSION_ID,
                            **identity,
                            "status": status,
                        },
                    )
                sequence += 1
                _send_observation(connection, sequence)
            elif message["type"] == "fallback":
                fallback_actions.append(message["payload"])
            elif message["type"] == "close":
                break
    finally:
        reader.close()
        connection.close()


def _send_observation(connection: socket.socket, sequence: int) -> None:
    _send(
        connection,
        {
            "type": "observation",
            "protocol_version": PROTOCOL_VERSION,
            "session_id": SESSION_ID,
            "sequence": sequence,
            "payload": {"joint": [sequence / 10]},
        },
    )


async def _serve(engine: AsyncServingEngine) -> tuple[Event, ...]:
    try:
        return await engine.run(0.35)
    finally:
        await engine.aclose()


def main() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        fallback_actions: list[object] = []
        robot = threading.Thread(target=_robot, args=(listener, fallback_actions))
        robot.start()
        host, port = listener.getsockname()
        transport = JsonlSocketTransport.connect(host, port, SESSION_ID)
        endpoint = RemoteEndpoint(
            transport,
            SessionConfig(SESSION_ID, control_hz=20, chunk_size=2),
            safe_action=lambda: {"joint": [0.0]},
        )
        engine = AsyncServingEngine(
            [endpoint],
            SyntheticBackend(chunk_size=2),
            FIFOScheduler(),
        )
        events = asyncio.run(_serve(engine))
        robot.join(timeout=1)
        if robot.is_alive():
            raise RuntimeError("robot server did not stop")
        if not fallback_actions or any(
            action != {"joint": [0.0]} for action in fallback_actions
        ):
            raise RuntimeError("robot server did not handle the expected safe fallback")
    actions = sum(event.kind == "action_executed" for event in events)
    observations = sum(event.kind == "observation_ready" for event in events)
    print(
        f"remote session: {observations} observations, {actions} executed actions, "
        f"{len(fallback_actions)} fallbacks handled"
    )


if __name__ == "__main__":
    main()
