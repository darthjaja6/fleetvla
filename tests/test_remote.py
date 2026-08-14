import asyncio
import json
import socket
import subprocess
import sys
import threading
import time

import pytest

from fleetvla import (
    ActionCommand,
    JsonlSocketTransport,
    RemoteEndpoint,
    SessionConfig,
)
from fleetvla.endpoints import ObservationUnavailable


def test_remote_serving_example_runs_end_to_end() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/remote_serving.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "remote session:" in completed.stdout
    assert "executed actions" in completed.stdout


def _send(peer, message):
    peer.sendall((json.dumps(message) + "\n").encode())


def test_jsonl_remote_endpoint_admits_session_and_exchanges_actions() -> None:
    client, server = socket.socketpair()
    received = []

    def robot() -> None:
        _send(
            server,
            {"type": "hello", "protocol_version": 2, "session_id": "arm"},
        )
        _send(
            server,
            {
                "type": "observation",
                "protocol_version": 2,
                "session_id": "arm",
                "sequence": 0,
                "payload": {"joint": [0.1, 0.2]},
            },
        )
        reader = server.makefile("r", encoding="utf-8")
        action = json.loads(reader.readline())
        received.append(action)
        identity = {
            key: action[key]
            for key in ("generation", "observation_sequence", "action_index")
        }
        for status in ("accepted", "executed"):
            _send(
                server,
                {
                    "type": "action_ack",
                    "protocol_version": 2,
                    "session_id": "arm",
                    **identity,
                    "status": status,
                },
            )
        for _ in range(3):
            received.append(json.loads(reader.readline()))
        reader.close()
        server.close()

    thread = threading.Thread(target=robot)
    thread.start()
    transport = JsonlSocketTransport(client, "arm")
    endpoint = RemoteEndpoint(
        transport,
        SessionConfig("arm", control_hz=20, chunk_size=2),
        safe_action=lambda: {"joint": [0.0, 0.0]},
    )

    for _ in range(100):
        try:
            observation = endpoint.observe()
            break
        except ObservationUnavailable:
            time.sleep(0.001)
    else:
        raise AssertionError("remote observation was not delivered")

    assert observation == {"joint": [0.1, 0.2]}
    deadline_s = time.monotonic() + 0.1
    receipt = asyncio.run(
        endpoint.execute_command(
            ActionCommand(
                "arm",
                generation=3,
                observation_sequence=7,
                action_index=4,
                value={"joint": [0.3, 0.4]},
                observation_captured_at_s=0,
                deadline_s=deadline_s,
            )
        )
    )
    endpoint.fallback()
    endpoint.close()
    thread.join(timeout=1)

    assert [message["type"] for message in received] == [
        "action",
        "fallback",
        "fallback",
        "close",
    ]
    assert all(message["session_id"] == "arm" for message in received)
    assert receipt.accepted is True and receipt.executed is True
    assert received[0]["generation"] == 3
    assert received[0]["observation_sequence"] == 7
    assert received[0]["action_index"] == 4
    assert 0 < received[0]["deadline_after_s"] <= 0.1


def test_jsonl_transport_rejects_wrong_session_admission() -> None:
    client, server = socket.socketpair()
    _send(
        server,
        {"type": "hello", "protocol_version": 2, "session_id": "alien"},
    )

    with pytest.raises(ValueError, match="admission"):
        JsonlSocketTransport(client, "arm")

    server.close()


def test_jsonl_transport_rejects_replayed_observation_sequence() -> None:
    client, server = socket.socketpair()
    _send(
        server,
        {"type": "hello", "protocol_version": 2, "session_id": "arm"},
    )
    observation = {
        "type": "observation",
        "protocol_version": 2,
        "session_id": "arm",
        "sequence": 4,
        "payload": {"joint": [0.1]},
    }
    _send(server, observation)
    _send(server, observation)
    transport = JsonlSocketTransport(client, "arm")

    for _ in range(100):
        try:
            transport.receive_observation()
        except ObservationUnavailable:
            time.sleep(0.001)
        except RuntimeError as error:
            assert "not increasing" in str(error)
            break
    else:
        raise AssertionError("replayed observation was not rejected")

    transport.close()
    server.close()


def test_remote_endpoint_rejects_unadmitted_fake_transport() -> None:
    class FakeTransport:
        protocol_version = 2
        session_id = "alien"

    with pytest.raises(ValueError, match="not admitted"):
        RemoteEndpoint(
            FakeTransport(),
            SessionConfig("arm", control_hz=20, chunk_size=2),
            safe_action=lambda: 0,
        )


def test_remote_action_requires_terminal_robot_acknowledgement() -> None:
    client, server = socket.socketpair()

    def robot() -> None:
        _send(
            server,
            {"type": "hello", "protocol_version": 2, "session_id": "arm"},
        )
        reader = server.makefile("r", encoding="utf-8")
        assert json.loads(reader.readline())["type"] == "action"
        time.sleep(0.03)
        assert json.loads(reader.readline())["type"] == "fallback"
        assert json.loads(reader.readline())["type"] == "close"
        reader.close()
        server.close()

    thread = threading.Thread(target=robot)
    thread.start()
    endpoint = RemoteEndpoint(
        JsonlSocketTransport(client, "arm"),
        SessionConfig("arm", control_hz=20, chunk_size=2),
        safe_action=lambda: 0,
        acknowledgement_timeout_s=0.01,
    )
    command = ActionCommand(
        "arm",
        generation=0,
        observation_sequence=0,
        action_index=0,
        value=1,
        observation_captured_at_s=0,
        deadline_s=time.monotonic() + 0.05,
    )

    with pytest.raises((TimeoutError, RuntimeError), match="remote"):
        asyncio.run(endpoint.execute_command(command))

    endpoint.close()
    thread.join(timeout=1)
