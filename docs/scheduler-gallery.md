# Scheduler gallery

The gallery groups algorithms by objective instead of declaring one winner.
Every entry must link its implementation, assumptions, source or contributor,
conformance result, and a versioned reproducible workload.

| Scheduler | Objective and assumptions | Source | Reproducible workload |
|---|---|---|---|
| FIFO | Oldest ready observation first; simple batching baseline | FleetVLA contributors | `benchmarks/contention.json` |
| Round robin | Rotate service across ready session IDs | Classical baseline | `benchmarks/contention.json` |
| EDF | Earliest action-buffer or request deadline | Classical real-time baseline | `benchmarks/contention.json` |
| Adaptive slack | Protect predicted buffer slack, batch while feasible | FleetVLA contributors; experimental | `benchmarks/batching.json` |

The published `contended-three-arm` seed-0 artifacts replay every event and
metric exactly:

| Scheduler | Useful actions | Starvation | p95 action age | Fairness | Artifact |
|---|---:|---:|---:|---:|---|
| FIFO | 105 | 65.0% | 0.700 s | 0.838 | [`JSON`](../benchmarks/results/contended-three-arm-fifo-s0.json) |
| Round robin | 104 | 65.3% | 0.700 s | 0.871 | [`JSON`](../benchmarks/results/contended-three-arm-round-robin-s0.json) |
| EDF | 109 | 63.7% | 0.600 s | 0.771 | [`JSON`](../benchmarks/results/contended-three-arm-edf-s0.json) |
| Adaptive slack | 109 | 63.7% | 0.600 s | 0.805 | [`JSON`](../benchmarks/results/contended-three-arm-adaptive-slack-s0.json) |

The table exposes the trade-off instead of naming a winner: EDF and adaptive
slack improve useful work and tail age here, round robin is fairer, and the
single-slot scenario cannot demonstrate batching gains.

Armory's Lookahead algorithm from [Action Chunk Scheduling for Batched Robot
Policy Serving](https://arxiv.org/abs/2608.00337) is an intended compatibility
target, not currently implemented. FleetVLA's scheduler contract can now defer
dispatch to coalesce near-future arrivals, which is one required building
block; Lookahead should only be added after its complete algorithm and
assumptions can be faithfully adapted and reproduced.

## System-track evidence

The checked-in
[`smolvla-libero-spatial-edf-rtx4090.json`](../benchmarks/results/smolvla-libero-spatial-edf-rtx4090.json)
is a measured 90-second integration run, not a scheduler leaderboard entry. It
uses all ten LIBERO Spatial tasks, the pinned public SmolVLA-LIBERO model, EDF,
prediction horizon 50, execution horizon 8, and one RTX 4090. The run completed
134 inferences at batch sizes 2, 3, and 4 with no serving failures, 3,445
policy actions, 80.9% starved or missed control ticks, and 2.805 s p95 action
age. The raw trace accounts for all 18,000 ticks configured across ten 20 Hz
sessions: 5,227 endpoint ticks ran and 12,773 were missed because the single
process did not sustain that aggregate cadence. Every task reached at least one
episode boundary; the run recorded 9 successes over 17 completed episodes.
This single-seed systems run is evidence for dynamic
batching, post-reset scheduling, full-suite protocol coverage, and end-to-end
behavior, not a standardized policy-quality or scheduler claim.

The system run predates the dispatch-deferral API and identifies source commit
`f8232ab` by its recorded source SHA-256. On newer source, inspect it with
`fleetvla verify-artifact --allow-source-mismatch` or check out that commit for
an exact source match. The flag verifies its checksum and internal consistency;
it does not claim that the historical GPU run exercised newer code.
