# Remote robot transport

`RemoteEndpoint` lets the wall-clock serving engine own a session whose robot
loop runs in another process or host. The included `JsonlSocketTransport` is a
TCP client with no third-party dependency. One connection carries exactly one
admitted session; it is not a discovery service or a safety controller.

Connect and construct the ordinary endpoint:

```python
from fleetvla import JsonlSocketTransport, RemoteEndpoint, SessionConfig

transport = JsonlSocketTransport.connect("robot-1.local", 7447, "arm-1")
endpoint = RemoteEndpoint(
    transport,
    SessionConfig("arm-1", control_hz=20, chunk_size=8),
    safe_action=lambda: {"joints": [0.0] * 7},
)
```

Run the complete standard-library localhost example before splitting the robot
and serving processes across hosts:

```bash
python examples/remote_serving.py
```

It starts a temporary TCP robot server, performs admission, streams increasing
observations, acknowledges each action as accepted and executed, exercises
fallback, and shuts both sides down. The example is protocol plumbing, not a
robot-side safety controller.

The robot server must first send one JSON line and wait for commands:

```json
{"type":"hello","protocol_version":2,"session_id":"arm-1"}
```

The version and session must exactly match the configured endpoint or admission
fails before the runtime registers the session. The robot then publishes
strictly increasing observations:

```json
{"type":"observation","protocol_version":2,"session_id":"arm-1","sequence":0,"payload":{"joints":[0.1,0.2]}}
```

FleetVLA sends a versioned command identity with every action:

```json
{"type":"action","protocol_version":2,"session_id":"arm-1","generation":0,"observation_sequence":3,"action_index":9,"deadline_after_s":0.05,"payload":{"joints":[0.2,0.3]}}
```

The robot must reject a command whose generation or action index is stale and
must consume it before the relative `deadline_after_s` budget expires. A
relative budget avoids assuming synchronized clocks across hosts. The robot
acknowledges the exact identity first as `accepted`,
and reports `executed` only after the local control loop consumes it:

```json
{"type":"action_ack","protocol_version":2,"session_id":"arm-1","generation":0,"observation_sequence":3,"action_index":9,"status":"accepted"}
{"type":"action_ack","protocol_version":2,"session_id":"arm-1","generation":0,"observation_sequence":3,"action_index":9,"status":"executed"}
```

The alternative terminal status is `rejected`, either directly after send or
after acceptance. Missing, duplicate, out-of-order, and mismatched
acknowledgements fail the endpoint. FleetVLA counts an action as useful only
after `executed`; TCP delivery alone is never robot execution. Configure the
ACK bound with `RemoteEndpoint(..., acknowledgement_timeout_s=...)` and the
total callback bound with `AsyncServingEngine(..., endpoint_timeout_s=...)`.

Fallback envelopes contain a `payload`; the final `close` envelope does not.
Every envelope includes `protocol_version` and `session_id`. Messages are UTF-8
JSON objects separated by `\n`; duplicate or backward observation sequences
close the usable stream. `RemoteEndpoint` validates observations and outbound
actions using the same schemas and callbacks as local endpoints.

TCP delivery does not provide authentication, encryption, a robot-side
deadline watchdog, or emergency stop. Put the connection behind an authenticated tunnel where
needed, and keep joint limits, liveliness, fallback, and stop authority in the
robot process. The public `RemoteTransport` protocol allows a deployment to
replace JSON-lines with a site-specific authenticated transport without
changing the scheduler or serving engine.
