# Benchmark interpretation

FleetVLA treats benchmarking as a way to understand a scheduler, not as a
single universal leaderboard. A run composes and records these independent
dimensions: scheduler and typed configuration, environment, backend latency
profile, fleet scenario, per-robot control and network timing, action-execution
policy, duration, and seed.

A backend may declare a linear latency model or an exact measured latency for
each supported batch size. Exact profiles are preferable when evaluating
dynamic batch-size algorithms because a fitted line can change their decisions.

`action_execution` is an explicit workload coordinate. `sequential-buffer`
finishes already buffered actions before a new chunk. `latest-indexed` assigns
each observation the next global action index; when its result arrives, actions
whose indices already executed are skipped and the remaining latest chunk
replaces older buffered predictions. Compare both when overlapping predictions
or closed-loop freshness matter. Neither label should be omitted from a result.

The synthetic and trace environments are deterministic: their `seed` is a
recorded experiment coordinate and changing it alone does not randomize request
arrivals. Integrations that use stochastic policies or environments are
responsible for consuming and recording the seed. Do not present repeated
synthetic seeds as independent trials.

The synthetic environment generates requests from each robot's live
action-buffer state. The trace environment instead replays recorded observation
arrival times from an artifact. The scheduler sees the same `FleetSnapshot` in
both environments and cannot access simulator rewards or privileged state.

Metrics have explicit units and denominators:

- `sent_actions`: commands handed to an endpoint transport;
- `accepted_actions`: commands accepted by the endpoint or explicitly
  acknowledged by a remote robot;
- `useful_actions`: actions actually consumed by local control loops;
- `starvation_frequency`: control ticks without an action, including scheduled
  ticks missed by a wall-clock engine, divided by all configured control ticks;
- `starvation_duration_s`: total robot-seconds without an action;
- `action_age_p50_s` and `action_age_p95_s`: observation capture to action
  execution, including inference, transport, and time buffered behind earlier
  actions;
- `fairness`: Jain's index over each robot's non-starved control-tick ratio;
- `batch_sizes`: one entry per inference dispatch;
- `backend_utilization`: modeled or measured inference time, according to the
  track, divided by run duration; and
- `per_session`: sent, accepted, and useful actions plus starvation, duration,
  and progress ratio for every robot.

For the in-process simulator, sending and acceptance are synchronous, so all
three action counts are equal. Remote serving keeps them separate: a timeout or
rejection can increase `sent_actions` without increasing `useful_actions`.
Historical version-1 system artifacts created before the acknowledged remote
protocol may omit the first two fields; verification treats their recorded
executions as the legacy synchronous send/accept path.

`dispatch_deferred` events record when a scheduler deliberately waited and the
absolute virtual- or monotonic-clock deadline it requested. Use them with batch
sizes and action age to evaluate whether coalescing helped or only added delay.
Wall-clock traces also record `scheduler_decision` latency and
`scheduler_failed` transitions to the EDF fallback.

Compare metric tables and timelines together. A scheduler may improve useful
work while making one robot less fresh, or improve fairness by using smaller
batches. Raw events in the checksummed JSON artifact are the evidence for those
trade-offs. The artifact also records FleetVLA, Python, platform, and local
scheduler source versions needed to interpret reproduction failures.

The artifact's `sha256` is computed over its canonical JSON body before the
`sha256` field is added. It protects semantic content independently of
whitespace and key formatting, so it is intentionally different from running
`sha256sum` on the formatted JSON file.

Use `benchmarks/contention.json` to emphasize deadline/fairness choices under a
single-slot backend. Use `benchmarks/batching.json` to expose the latency versus
batch-size decision with two batch slots. The
`benchmarks/armory-one-fast-l40s.json` systems proxy fixes the paper's published
L40S latency profile, heterogeneous horizons, and `latest-indexed` execution
for a controlled RR/EDF/Lookahead comparison. None is a universal score.

The optional system track uses the same scheduler with measured policy and
simulator latency:

```bash
MUJOCO_GL=egl fleetvla libero --scheduler edf --task 0 --task 1 \
  --execution-horizon 8 --action-execution latest-indexed \
  --output libero-system-result.json
```

System artifacts use `artifact_kind: "system"`. They record raw events and a
checksum like algorithm artifacts. Verify either artifact kind without executing
embedded scheduler code or attempting a replay:

```bash
fleetvla verify-artifact libero-system-result.json
```

Verification requires the recorded FleetVLA source SHA-256 to match the
installed source. Use `--allow-source-mismatch` only to inspect a historical
artifact; it does not make that artifact evidence for the installed build.
For system artifacts, verification also recomputes serving metrics and LIBERO
task outcomes from the typed raw event trace.
The checksum establishes integrity and internal consistency, not independent
authenticity of a measurement; trusted attestation would require an external
signing or logging system.

Each artifact includes a SHA-256 identity over the installed FleetVLA Python
source, which distinguishes development builds that share the same package
version. System artifacts are not replayed bit-for-bit: GPU kernels, wall-clock
scheduling, and physics are not deterministic virtual time. Report task success
only for runs long enough to reach valid episode boundaries; the default
three-second command is an integration smoke benchmark.
