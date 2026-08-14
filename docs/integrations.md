# Policy and simulator integrations

FleetVLA keeps four contracts separate: an endpoint produces observations and
executes actions, a backend turns a dynamic batch into versioned action chunks,
a scheduler chooses ready session IDs from frozen state, and an environment
adapter retains task-specific rewards and success signals. The scheduler code
does not change when any of the other three change.

Remote robot processes use the same endpoint contract through the
[versioned JSON-lines transport](remote.md). Admission binds one connection to
one configured session before its observations can enter the runtime; action,
fallback, validation, and disconnect behavior remain endpoint-local. A remote
action becomes useful only after the robot acknowledges acceptance and actual
control-loop execution.

## SmolVLA and LeRobot policies

The optional backend follows LeRobot 0.6.2's public
`PreTrainedPolicy.predict_action_chunk` contract: a mapping of tensors with a
batch axis goes in and a tensor shaped `[batch, horizon, action_dim]` comes out.
Install it on Python 3.12 or newer:

```bash
python -m pip install -e '.[smolvla]'
```

The extra installs the exact LeRobot commit named below because its 0.6.2
source API is newer than the latest PyPI release. Git is therefore required for
this optional installation; the dependency-free core does not require it.

For LIBERO, its upstream EGL probes currently need CMake available while pip
builds them and do not declare that build requirement themselves. Bootstrap
those two source packages before installing the extra:

```bash
python -m pip install 'cmake>=3.29,<4' wheel setuptools
python -m pip install --no-build-isolation egl-probe==1.0.2 hf-egl-probe==1.0.2
python -m pip install -e '.[smolvla,libero]'
```

CMake 4 is excluded because `egl-probe==1.0.2` still uses pre-3.5 CMake policy
compatibility. This bootstrap is specific to the optional simulator stack.

Load a published SmolVLA policy and its matching pre/postprocessors:

```python
from fleetvla.integrations import LeRobotPolicyBackend

backend = LeRobotPolicyBackend.from_smolvla_pretrained(
    "lerobot/smolvla_base",
    revision="PIN_A_MODEL_REVISION_FOR_RESULTS",
)
```

The model output horizon and robot execution horizon are distinct. By default,
the backend respects LeRobot's `n_action_steps`; an explicit
`execution_horizon` may select a longer prefix for action-chunking experiments,
but it cannot exceed the model output. Each endpoint's `chunk_size` must equal
that execution horizon. A mismatch is rejected before any action reaches the endpoint. The
backend batches environment-processed observations before invoking the policy
processor once, so tokenization and padding see the complete dynamic batch. It
records the policy type and output shape on each chunk and updates its latency
cost estimate from measured inference. Model weights and tensors never enter
`SessionSnapshot`.

This integration was checked against LeRobot commit
`a16f34c085c9597fcbdb9fde395a3334d78df716` (0.6.2). See the upstream
[policy API](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/pretrained.py)
and [SmolVLA implementation](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/smolvla/modeling_smolvla.py).

## LIBERO parallel simulation

`LiberoVectorAdapter` supports both LIBERO's indexed
`BaseVectorEnv.reset(id=...)` / `step(action, id=...)` interface and LeRobot's
current Gymnasium vector interface. Use the default `addressing="indexed"` to
expose slots of an indexed vector environment. Use `addressing="single"` for a
one-slot Gymnasium vector environment; create one such environment per session
so their control loops can advance independently. Rewards, termination, and
success stay in `adapter.metrics`; they are deliberately absent from the
scheduler snapshot.
When a slot terminates or truncates, its endpoint reports an episode boundary
to the serving engine. The engine acknowledges the final action, increments the
session generation, discards the remainder of the old chunk, clears opaque
policy state before the next inference batch, and observes the reset episode.
`LeRobotPolicyBackend` requires a resettable policy and treats its caches as
batch-local: the first inference and every inference after a session reset call
`policy.reset()`. Use `StatefulPolicyBackend` when a model needs genuinely
per-session recurrent state. For LIBERO's four-value API,
`done` is task success. For LeRobot's five-value LIBERO API, task success comes
from `info["is_success"]` (`success` is accepted as a compatibility alias);
termination and truncation are recorded separately and are not inferred to be
success.

```python
from fleetvla import SessionConfig, create_scheduler
from fleetvla.integrations import LiberoVectorAdapter
from fleetvla.serving import AsyncServingEngine, serving_metrics

adapter = LiberoVectorAdapter(
    libero_vector_env,
    [
        SessionConfig("task-0", 20, 8),
        SessionConfig("task-1", 10, 8),
    ],
    observation_converter=to_lerobot_observation,
    action_converter=to_libero_action,
    fallback_action=lambda: zero_action,
)
engine = AsyncServingEngine(
    adapter.endpoints,
    backend,
    create_scheduler("adaptive-slack"),
    max_batch_size=2,
    scheduler_timeout_s=0.01,
    action_execution="latest-indexed",
)
events = asyncio.run(engine.run(30.0))
system_metrics = serving_metrics(events, adapter.endpoints, 30.0)
task_metrics = adapter.metrics
```

The repository provides the complete current LeRobot path as a system-track
command. Its default public model and revision are pinned; repeat `--task` to
choose independent sessions and use any registered or local scheduler:

```bash
MUJOCO_GL=egl fleetvla libero --scheduler edf \
  --task 0 --task 1 --duration 3 --execution-horizon 8 \
  --scheduler-timeout 0.01 \
  --output libero-system-result.json
```

The command writes raw events, system and task metrics, full configuration,
software versions, accelerator identity, and a SHA-256 checksum. It intentionally
does not offer deterministic replay for a wall-clock GPU/simulator run. Omitting
`--execution-horizon` preserves the model's configured `n_action_steps`; an
override is recorded and should be treated as an experiment variable when task
quality is compared.

Verify the schema, result checksum, exact FleetVLA source identity, and any
embedded local-scheduler source hash without rerunning physics or executing code:

```bash
fleetvla verify-artifact libero-system-result.json
```

Use LIBERO's `SubprocVectorEnv` when physics parallelism is needed. FleetVLA
does not own process creation or simulator reset semantics. This adapter was
checked against LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` and its
[vector environment](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/venv.py).

## Validation status

| Surface | Repository evidence | Not yet verified here |
|---|---|---|
| Core synthetic/trace serving | Executed CPU tests and CLI journeys | — |
| LeRobot policy backend | Fake-policy tests plus SmolVLA-LIBERO revision `6721902...` on an RTX 4090; measured dynamic batches up to `[4,50,7]` | FastWAM inference: public 26.24 GB snapshot downloaded and config parsed, but policy load exceeded this 31 GB RAM/24 GB GPU host |
| LIBERO adapter | Fake protocol tests plus all ten real `libero_spatial` Gymnasium environments through the serving engine | Standardized multi-seed task-quality evaluation |
| ROS 2 bridge | Fake node/publisher/subscription protocol tests | A ROS 2 distribution and live topics |
| LeRobot robot endpoint | Fake driver and failure/safety tests | Physical robot hardware |
| GPU container | Dockerfile/Compose static checks | Docker/NVIDIA runtime and GPU execution |

The published 90-second run completed 134 inferences at batch sizes 2, 3, and 4
without serving failures. All ten tasks reached a configured episode boundary.
It also reports 12,773 missed scheduled ticks instead of silently excluding the
fact that one process did not sustain ten 20 Hz environments. This single-seed
systems run is not a standard LIBERO policy evaluation.
Integration pull requests must keep this table or their own validation statement
equally explicit.

The synthetic and trace environments use virtual time and complete without
sleeping. `AsyncServingEngine` uses a monotonic wall clock; endpoint control
ticks and fallbacks continue while policy inference runs in a worker thread.
An optional inference timeout resets the selected generations and invokes local
fallback. Python cannot kill a running inference thread, so the engine
quarantines that worker and will not enter the backend again until it exits;
this preserves single-entry policy/model semantics instead of pretending the
GPU call was cancelled.
Benchmark artifacts must state which clock and integration were used. Do not
compare modeled latency with measured GPU latency as though they were the same.
