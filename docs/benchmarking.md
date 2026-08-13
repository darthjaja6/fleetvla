# Benchmark interpretation

FleetVLA treats benchmarking as a way to understand a scheduler, not as a
single universal leaderboard. A run composes and records these independent
dimensions: scheduler and typed configuration, environment, backend latency
profile, fleet scenario, per-robot control and network timing, action-execution
policy, duration, and seed.

The synthetic environment generates requests from each robot's live
action-buffer state. The trace environment instead replays recorded observation
arrival times from an artifact. The scheduler sees the same `FleetSnapshot` in
both environments and cannot access simulator rewards or privileged state.

Metrics have explicit units and denominators:

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
- `per_session`: useful actions, starvation, duration, and progress ratio for
  every robot.

Compare metric tables and timelines together. A scheduler may improve useful
work while making one robot less fresh, or improve fairness by using smaller
batches. Raw events in the checksummed JSON artifact are the evidence for those
trade-offs. The artifact also records FleetVLA, Python, platform, and local
scheduler source versions needed to interpret reproduction failures.

Use `benchmarks/contention.json` to emphasize deadline/fairness choices under a
single-slot backend. Use `benchmarks/batching.json` to expose the latency versus
batch-size decision with two batch slots. Neither is a universal score.

The optional system track uses the same scheduler with measured policy and
simulator latency:

```bash
MUJOCO_GL=egl fleetvla libero --scheduler edf --task 0 --task 1 \
  --execution-horizon 8 \
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
