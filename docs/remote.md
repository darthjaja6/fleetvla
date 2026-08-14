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

The robot server must first send one JSON line and wait for commands:

```json
{"type":"hello","protocol_version":1,"session_id":"arm-1"}
```

The version and session must exactly match the configured endpoint or admission
fails before the runtime registers the session. The robot then publishes
strictly increasing observations:

```json
{"type":"observation","protocol_version":1,"session_id":"arm-1","sequence":0,"payload":{"joints":[0.1,0.2]}}
```

FleetVLA sends `action` and `fallback` envelopes with a `payload`, and a final
`close` envelope without one. Every envelope includes `protocol_version` and
`session_id`. Messages are UTF-8 JSON objects separated by `\n`; duplicate or
backward observation sequences close the usable stream. `RemoteEndpoint`
validates observations and outbound actions using the same schemas and
callbacks as local endpoints.

TCP delivery does not provide authentication, encryption, a deadline watchdog,
or emergency stop. Put the connection behind an authenticated tunnel where
needed, and keep joint limits, liveliness, fallback, and stop authority in the
robot process. The public `RemoteTransport` protocol allows a deployment to
replace JSON-lines with a site-specific authenticated transport without
changing the scheduler or serving engine.
