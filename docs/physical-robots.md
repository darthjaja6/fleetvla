# Physical endpoints and local safety

The central scheduler is not a safety controller. Network loss, a process
pause, an invalid action, or an empty buffer must be handled at the robot.
FleetVLA's `LeRobotRobotEndpoint` wraps the current LeRobot robot interface:
`get_observation`, `send_action`, and `disconnect`.

Construct it with an explicit action converter, safe fallback, and validator:

```python
endpoint = LeRobotRobotEndpoint(
    robot,
    SessionConfig("so101", control_hz=30, chunk_size=8),
    observation_schema=(FieldSpec("observation.state", (6,)),),
    action_converter=policy_vector_to_robot_dict,
    safe_action=hold_current_position,
    action_validator=within_joint_and_velocity_limits,
)
```

The default validator only rejects non-finite values; it is not enough for a
real robot. Supply joint, velocity, workspace, collision, and calibration checks
appropriate to the platform. `fallback` is called on starvation and inference
failure. An execution exception disconnects the session, increments its
generation, clears buffered actions, and closes the endpoint. Old in-flight
chunks and acknowledgements are then rejected.

The wall-clock engine runs each endpoint's observe, execute, fallback, and
close callbacks outside the shared asyncio loop and applies
`endpoint_timeout_s`. Sessions have independent locks, so a slow driver cannot
run concurrently with its own close/reconnect or block a healthy peer. Python
cannot terminate a native driver call safely after timeout; hardware adapters
must still configure finite device or middleware deadlines.
Before destroying a shared driver or simulator environment, call
`await engine.aclose()` in the same event loop that ran the engine. It waits for
timed-out endpoint callbacks before closing endpoints. Synchronous endpoints may
use `engine.close()` after `run()` returns; it serializes shutdown against any
worker thread that outlived its callback timeout.

`ROS2Endpoint` uses a node's standard `create_subscription` and
`create_publisher` methods. Application code supplies message types and
converters because observation and command messages are robot-specific. A safe
message is published on close or starvation. Keep the ROS 2 controller's own
deadline, liveliness, watchdog, and joint-limit enforcement enabled; the bridge
does not replace them.
Each subscription sample is consumed at most once. Until another callback
arrives, `observe()` reports `ObservationUnavailable`, so a stopped topic cannot
create new action chunks from one cached sensor message.

Reconnect is explicit. A LeRobot endpoint calls the driver's `connect` before
the runtime marks the new generation connected. A ROS 2 endpoint clears its
cached message and remains observation-unavailable until a post-reconnect
sample arrives; it never reuses the last pre-disconnect observation. Both
endpoint types reject execution after close. ROS 2 commands use the same
non-finite default rejection as LeRobot, and should receive a stricter
platform-specific validator in production. ROS validation runs after conversion
on the exact outbound message, and also validates the fallback message before
publishing it.

Before hardware:

1. run the endpoint with a fake driver and intentional NaN, timeout, disconnect,
   reset, duplicate acknowledgement, and late-chunk cases;
2. run against a simulator with the same converters;
3. test at low speed with an independent emergency stop;
4. verify fallback locally while the FleetVLA process is killed; and
5. record calibration, policy, scheduler, schema, control-rate, and safety-limit
   versions with the deployment result.

No physical hardware was used for the repository tests. They verify protocol
and failure behavior with fake LeRobot and ROS 2 objects, not robot safety.
