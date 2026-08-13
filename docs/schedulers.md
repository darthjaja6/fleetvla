# Write and evaluate a scheduler

A FleetVLA scheduler answers one narrow question: which ready sessions should
share the next inference call? The runtime owns mutable robot state. Your
scheduler receives a frozen `FleetSnapshot` plus an `InferenceCostModel`, and
returns a `ScheduleDecision` containing session IDs.

Start with the complete example in `examples/my_scheduler.py`. It sorts ready
robots by remaining executable action time:

```python
from fleetvla import FleetSnapshot, InferenceCostModel, ScheduleDecision

class SmallestBufferFirst:
    def schedule(self, fleet: FleetSnapshot, costs: InferenceCostModel):
        ready = sorted(
            fleet.ready_sessions,
            key=lambda session: (session.buffer_horizon_s, session.session_id),
        )
        return ScheduleDecision(
            tuple(s.session_id for s in ready[:fleet.max_batch_size]),
            "smallest action buffer first",
        )
```

No installation hook or runtime edit is required. Check the contract directly:

```bash
fleetvla test-scheduler ./examples/my_scheduler.py:SmallestBufferFirst
```

Then run it on the versioned heterogeneous workload:

```bash
fleetvla benchmark \
  --config benchmarks/heterogeneous.json \
  --scheduler ./examples/my_scheduler.py:SmallestBufferFirst \
  --output results/custom.json \
  --timeline
```

The artifact records the full scheduler, environment, backend, fleet, timing,
action-execution, and seed configuration alongside raw events and metrics. A
local-scheduler artifact embeds executable source, so inspect
`provenance.scheduler_source` and explicitly opt in when replaying it:

```bash
fleetvla replay results/custom.json --allow-embedded-scheduler
```

This reruns it and compares every event and metric. Use an existing artifact's
recorded observation times as a trace-driven workload with another scheduler:

```bash
fleetvla benchmark --environment trace --trace results/custom.json \
  --scheduler edf --output results/edf-from-trace.json
```

This counterfactual trace run holds observation arrivals fixed. If the new
scheduler still has an in-flight request at a recorded arrival, the runtime
records `observation_dropped_backpressure`. This differs from `replay`, which
reruns the original environment and requires every event and metric to match.
The derived artifact embeds the normalized arrival schedule, so it remains
replayable if it is moved or the source artifact is removed.

A local-scheduler artifact also embeds the exact single-file source and its
hash. Without the opt-in flag, FleetVLA refuses to execute code embedded in
JSON. Built-in scheduler artifacts never require the flag.

## What the snapshot means

`buffer_horizon_s` is executable actions divided by control frequency; compare
it with predicted inference and transport time, not with action count.
`request_time_s` timestamps the ready observation. `in_flight_sequence` and
`connected` distinguish backpressure and endpoint state, although only ready
sessions can be selected. Generations and sequence numbers remain runtime
responsibilities: a scheduler never constructs or accepts chunks.

Built-ins live in `fleetvla/schedulers/` and are registered explicitly in that
package's `__init__.py`: FIFO, round robin, earliest deadline first, and adaptive
slack. Algorithm-specific configuration is validated before a run. For example:

```bash
fleetvla benchmark --scheduler adaptive-slack \
  --scheduler-config '{"transport_margin_s": 0.02, "batch_size_limit": 2}'
```

When contributing a built-in algorithm, include its source paper or rationale,
typed configuration, conformance test, a deterministic benchmark, and an
explanation of the trade-off visible in the artifact. Do not import simulator,
robot, reward, or policy framework types into scheduler code.
