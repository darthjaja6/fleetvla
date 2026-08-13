# Contributing to FleetVLA

FleetVLA welcomes scheduling algorithms, workload definitions, and focused
backend or endpoint adapters. Start by running the CPU path:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
fleetvla demo
```

For a scheduler, first follow [the scheduler guide](docs/schedulers.md). A pull
request should keep the scheduler independent of runtime and environment
internals, pass `fleetvla test-scheduler`, and attach or describe a reproducible
benchmark artifact. Cite the originating paper or implementation and preserve
its license and attribution. Results should report the full workload rather
than a context-free scalar score.

For runtime or adapter changes, include tests for ordering, time units, stale
generations, resets, disconnects, buffer bounds, and deterministic virtual-time
behavior where relevant. Physical endpoints must retain local watchdogs and
safety limits; the central scheduler is not a safety controller.

Keep changes reviewable and avoid unrelated refactors or new dependencies.
Public contract changes need a design discussion before implementation. During
the pre-alpha period, compatibility is documented release by release; once a
stable contract is declared, removals will require a deprecation period.

Use the repository issue forms before substantial scheduler, backend, or
endpoint work. A scheduler contribution should also update the
[scheduler gallery](docs/scheduler-gallery.md). Integration pull requests must
state the tested upstream revision and distinguish fake/protocol, simulator,
and physical validation. See the [compatibility policy](docs/compatibility.md)
and [security policy](SECURITY.md).
