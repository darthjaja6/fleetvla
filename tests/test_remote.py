import json
import socket
import threading
import time

import pytest

from fleetvla import JsonlSocketTransport, RemoteEndpoint, SessionConfig
from fleetvla.endpoints import ObservationUnavailable


def _send(peer, message):
    peer.sendall((json.dumps(message) + "\n").encode())


def test_jsonl_remote_endpoint_admits_session_and_exchanges_actions() -> None:
    client, server = socket.socketpair()
    received = []

    def robot() -> None:
        _send(
            server,
            {"type": "hello", "protocol_version": 1, "session_id": "arm"},
        )
        _send(
            server,
            {
                "type": "observation",
                "protocol_version": 1,
                "session_id": "arm",
                "sequence": 0,
                "payload": {"joint": [0.1, 0.2]},
            },
        )
        reader = server.makefile("r", encoding="utf-8")
        for _ in range(4):
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
    endpoint.execute({"joint": [0.3, 0.4]})
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


def test_jsonl_transport_rejects_wrong_session_admission() -> None:
    client, server = socket.socketpair()
    _send(
        server,
        {"type": "hello", "protocol_version": 1, "session_id": "alien"},
    )

    with pytest.raises(ValueError, match="admission"):
        JsonlSocketTransport(client, "arm")

    server.close()


def test_jsonl_transport_rejects_replayed_observation_sequence() -> None:
    client, server = socket.socketpair()
    _send(
        server,
        {"type": "hello", "protocol_version": 1, "session_id": "arm"},
    )
    observation = {
        "type": "observation",
        "protocol_version": 1,
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
        protocol_version = 1
        session_id = "alien"

    with pytest.raises(ValueError, match="not admitted"):
        RemoteEndpoint(
            FakeTransport(),
            SessionConfig("arm", control_hz=20, chunk_size=2),
            safe_action=lambda: 0,
        )
