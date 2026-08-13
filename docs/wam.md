# WAM compatibility boundary

World-action models still need action chunking: the robot consumes a finite
action horizon while the next inference request is scheduled and served. The
current LeRobot FastWAM policy exposes the same
`predict_action_chunk -> [batch, action_horizon, action_dim]` surface as
SmolVLA, so the ordinary `LeRobotPolicyBackend` path can serve its actions.

FleetVLA intentionally does not make every policy use a multi-stage world-model
graph. The common runtime operation remains one batch producing one versioned
action chunk. `ActionChunk.auxiliary` can carry typed diagnostic summaries, and
`StatefulPolicyBackend` keeps opaque per-session recurrent or world state out of
the scheduler. Reset and disconnect increment a per-session invalidation token;
the token is bound before asynchronous worker handoff, and next state is committed
only after the corresponding generation-tagged chunk is accepted. An inference
that finishes afterward cannot restore old state. A scheduler may later see
small declared summaries such as cache cost, but never raw videos, latent
tensors, or simulator privileged state.

The compatibility spike covers:

- variable model-internal state isolated by session;
- action chunks and auxiliary world-model metadata in one result;
- an execution-horizon prefix independent of the model's prediction horizon;
- state reset on endpoint reset/disconnect; and
- unchanged FIFO, EDF, round-robin, or adaptive-slack inputs.

It does not yet cover separately scheduled video generation, cross-session KV
cache reuse, multi-stage admission, or partial world/action results. Those need
measured FastWAM workloads before the stable runtime contract expands. The
reference inspected was LeRobot's
[FastWAM implementation](https://github.com/huggingface/lerobot/tree/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/fastwam)
at commit `a16f34c085c9597fcbdb9fde395a3334d78df716`.

A real checkpoint-load attempt used the public
[`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224)
snapshot `a784645f9ce367ba6953dd20a7a7f0310c85747c`. Its LeRobot config loaded and
declared a 32-step action horizon with two 224-pixel camera inputs, 8-dimensional
state, and 7-dimensional actions. The complete 26.24 GB snapshot downloaded,
but `FastWAMPolicy.from_pretrained` was killed while loading on this 31 GB RAM,
24 GB RTX 4090 host before FleetVLA received a policy object. No FastWAM
inference or FleetVLA system result is therefore claimed. This is a measured
host-capacity boundary, not evidence that a new scheduling contract is needed;
the upstream reproduction recipe uses a 140 GB H20.
