# FleetVLA

FleetVLA is simulator-first infrastructure for serving action-chunking robot
policies to heterogeneous fleets. It separates scheduling policy from session
lifecycle, inference, and robot or simulator integration, so researchers can
compare a scheduling idea and then run the same scheduler in a real serving
path.

The project is under active development. The current runnable system has a
deterministic discrete-event simulator, stateful sessions, stale-result
rejection, a batch-aware synthetic backend, structured traces, five schedulers,
direct local scheduler loading, and reproducible benchmark artifacts. Optional
policy, parallel simulator, wall-clock serving, LeRobot robot, and ROS 2
adapters reuse the same scheduler contract; physical validation remains
platform-specific.

## Try it

FleetVLA has no runtime dependencies. With Python 3.10 or newer:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
fleetvla demo
```

The demo advances virtual time rather than sleeping, so it is deterministic and
runs on a CPU in a fraction of a second. Its 20 Hz and 10 Hz arms share one
synthetic backend; the output reports per-robot progress and dispatch batch
sizes.

Abbreviated output:

```text
FleetVLA deterministic CPU demo
fast-arm: 40 actions, 0 starved ticks
slow-arm: 20 actions, 0 starved ticks
dispatch batch sizes: [2, 1, 2, 1, ...]
```

If `python3 -m venv` reports that `ensurepip` is unavailable, install your
distribution's venv package first (for example `python3-venv` on Debian or
Ubuntu), then recreate `.venv`. After activation, check `python --version`;
FleetVLA requires Python 3.10 or newer.

Without administrator access, an environment manager that bundles its own pip
seeder is also sufficient. If neither `pip` nor `uv` is installed, this complete
non-admin path keeps the tool, cache, and managed-Python metadata inside the
checkout:

```bash
mkdir -p .tools .uv-cache .uv-python
curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_UNMANAGED_INSTALL="$PWD/.tools" sh
UV_CACHE_DIR="$PWD/.uv-cache" \
UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" \
  .tools/uv venv --seed --clear .venv
. .venv/bin/activate
python -m pip install -e .
```

The installer requires network access and `curl`; use your distribution's
packaged `uv` instead when available. `--seed` installs pip, and `--clear`
replaces the partial environment left by a failed `python3 -m venv` attempt.

## Lifecycle in concrete terms

Suppose two robots publish observations while one GPU is idle. The runtime
records each observation and exposes immutable session snapshots to a
scheduler. The scheduler returns session IDs to batch, or a future dispatch
time when it wants to wait briefly for more work; it cannot mutate robot state.
The backend produces versioned action chunks while robot control ticks continue
independently. On delivery, the runtime accepts only the chunk that
matches the session's current generation and request. Each robot consumes its
own action buffer at its configured control rate. A reset or disconnect bumps
the generation, so an old GPU result cannot reach the robot.

The simulator uses exactly this lifecycle. A backend's latency increases with
batch size, each robot can have a different control rate and network delay, and
all state changes are emitted as structured events.

## Try scheduling ideas

Compare all built-in algorithms on a three-arm workload with two batch slots:

```bash
fleetvla benchmark --config benchmarks/batching.json \
  --scheduler fifo --scheduler round-robin --scheduler edf \
  --scheduler adaptive-slack --scheduler lookahead \
  --output results --timeline
```

The comparison reports useful actions, starvation, action age, fairness, and
batch sizes. Every JSON artifact includes the complete configuration, raw event
trace, multi-dimensional metrics, and a checksum. Reproduce one with:

```bash
fleetvla replay results/batching-three-arm-fifo-s0.json
```

Use `benchmarks/contention.json` separately for a single-slot ordering and
fairness stress test; it cannot demonstrate batching gains.

To implement and load a scheduler without modifying FleetVLA, follow the
[scheduler guide](docs/schedulers.md). The path from a 15-line local class to
conformance testing and a benchmark is the primary contributor interface.

## Connect policies and environments

The same scheduler object runs in the deterministic synthetic/trace lab or the
wall-clock `AsyncServingEngine`. The optional `LeRobotPolicyBackend`
dynamically batches SmolVLA-style action chunks; `LiberoVectorAdapter` exposes
parallel simulator slots without leaking rewards or privileged state into
scheduler inputs. LeRobot robot and ROS 2 endpoints keep conversion and fallback
at the local execution boundary.

For a robot process on another host, `RemoteEndpoint` and
`JsonlSocketTransport` provide a versioned one-session admission handshake and
newline-delimited observation/action/fallback protocol using only the standard
library. See the [remote transport guide](docs/remote.md).

Start with the [integration guide](docs/integrations.md), then read the
[physical endpoint safety guide](docs/physical-robots.md) before connecting
hardware. The [WAM note](docs/wam.md) explains why action-only FastWAM fits the
current backend and which multi-stage capabilities are deferred. GPU/container
setup is in [the deployment recipe](docs/deployment/gpu.md).

With the optional simulator stack installed, one command runs the same public
scheduler contract against two real SmolVLA/LIBERO sessions and writes a
checksummed system-track artifact:

```bash
MUJOCO_GL=egl fleetvla libero --scheduler edf \
  --task 0 --task 1 --execution-horizon 8 \
  --action-execution latest-indexed \
  --output libero-system-result.json
```

This is a measured wall-clock run, not a deterministic replay. The artifact
records the pinned model revision, environment tasks, software and GPU versions,
raw serving events, system metrics, and task outcomes.

## Develop

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

FleetVLA is Apache-2.0 licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) for
validation, attribution, compatibility, and safety expectations. The public
surface remains pre-alpha.

## Related work

The paper [Action Chunk Scheduling for Batched Robot Policy
Serving](https://arxiv.org/abs/2608.00337) and its
[Armory implementation](https://github.com/GaTech-RL2/armory) established the
shared-GPU action-chunk scheduling problem and introduced Lookahead alongside
classical baselines. FleetVLA's built-in `lookahead` is an attributed L=1
adaptation over its immutable snapshot and includes a controlled systems proxy
using the paper's ten-session horizons and published L40S latency profile.
FleetVLA does not claim those ideas as its invention. Armory remains the more
complete reference for the paper's OpenPI, GR00T, and LIBERO evaluation path.
