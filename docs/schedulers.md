# Write and evaluate a scheduler

A FleetVLA scheduler answers one narrow question: which ready sessions should
share the next inference call? The runtime owns mutable robot state. Your
scheduler receives a frozen `FleetSnapshot` plus an `InferenceCostModel`, and
returns a `ScheduleDecision` containing session IDs. A scheduler may instead
return an empty selection with an absolute `defer_until_s` when waiting briefly
could form a better batch.

The snapshot also declares `action_execution`: `sequential-buffer` preserves
the old queue before a returned chunk, while `latest-indexed` replaces it and
drops prediction indices already consumed during inference. A scheduler that
predicts future chunk utility should model this field; ordering-only schedulers
can ignore it.

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
            tuple(s.session_id for s in ready[: fleet.max_batch_size]),
            "smallest action buffer first",
        )
```

No installation hook or runtime edit is required. Check the contract directly:

```bash
fleetvla test-scheduler ./examples/my_scheduler.py:SmallestBufferFirst
```

To coalesce near-simultaneous arrivals, defer from the oldest request rather
than repeatedly adding a delay to `fleet.now_s`:

```python
oldest_request_s = min(s.request_time_s for s in fleet.ready_sessions)
dispatch_at_s = oldest_request_s + 0.010
if fleet.now_s < dispatch_at_s:
    return ScheduleDecision(
        (), "wait up to 10 ms for peers", defer_until_s=dispatch_at_s
    )
```

At the deadline the runtime calls the scheduler again with a fresh snapshot.
It also reevaluates earlier if the ready set changes. Robot control ticks and
local fallbacks continue during the wait. A deferral must be finite and later
than `fleet.now_s`; `fleetvla test-scheduler` checks that contract.

`schedule()` remains an ordinary synchronous method, but the wall-clock engine
runs it on a dedicated daemon worker while control ticks continue. Each decision
has a 10 ms default budget (`scheduler_timeout_s`); a timeout, exception, or
invalid decision disables that worker and switches the run to EDF. The raw trace
records `scheduler_decision` latency and any `scheduler_failed` transition.
Keep computation bounded, return promptly, and use `defer_until_s` rather than
sleeping. This isolates serving timing and failure for trusted plugins; it is
not a security sandbox for untrusted Python. Deterministic simulator benchmarks
call schedulers directly and therefore have no wall-clock timeout.

The conformance command runs two sequential ready-set shapes through the same
scheduler state, repeats that sequence on a fresh instance, and executes a
decision through the wall-clock worker's 10 ms budget. This catches fixture-ID
assumptions, non-deterministic state transitions, and plugins that would
immediately fall back in serving.

Then run it on the versioned batching workload, which exposes batch-size choices:

```bash
fleetvla benchmark \
  --config benchmarks/batching.json \
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

Also run `benchmarks/contention.json` for ordering and fairness under a
single-slot backend. `benchmarks/heterogeneous.json` remains a quick integration
check, but similar policies often tie on it. Use
`benchmarks/armory-one-fast-l40s.json` when an algorithm models weighted
execution horizons and dynamic batch size. Report every trade-off, including a
tie or regression.

A local-scheduler artifact also embeds the exact single-file source and its
hash. Without the opt-in flag, FleetVLA refuses to execute code embedded in
JSON. Built-in scheduler artifacts never require the flag.

Replayable local schedulers must be self-contained in that one file and import
only the Python standard library and FleetVLA. Artifact replay does not capture
local helper modules or third-party package versions. If an experiment needs
those dependencies, record and distribute its environment separately; the
artifact alone is not a portable reproduction.

## What the snapshot means

`buffer_horizon_s` is executable actions divided by control frequency; compare
it with predicted inference and transport time, not with action count.
`request_time_s` timestamps the ready observation. `chunk_size` is the
session's execution horizon, while `service_weight` expresses an explicit
operator priority such as Armory's fast-tier weight. `in_flight_sequence` and
`connected` distinguish backpressure and endpoint state, although only ready
sessions can be selected. Generations and sequence numbers remain runtime
responsibilities: a scheduler never constructs or accepts chunks.

`InferenceCostModel` accepts either a linear estimate
(`base_latency_s + per_item_latency_s * batch_size`) or a measured latency value
for every supported batch size. The Armory-compatible workload uses the paper's
published L40S profile rather than fitting it to a line.

Built-ins live in `fleetvla/schedulers/` and are registered explicitly in that
package's `__init__.py`: FIFO, round robin, earliest deadline first, adaptive
slack, and the attributed L=1 Lookahead adaptation. Algorithm-specific
configuration is validated before a run. For example:

```bash
fleetvla benchmark --scheduler adaptive-slack \
  --scheduler-config '{"transport_margin_s": 0.02, "batch_size_limit": 2}'
```

When contributing a built-in algorithm, include its source paper or rationale,
typed configuration, conformance test, a deterministic benchmark, and an
explanation of the trade-off visible in the artifact. Do not import simulator,
robot, reward, or policy framework types into scheduler code. Report scheduler
decision cost for algorithms whose search is not trivially bounded.
