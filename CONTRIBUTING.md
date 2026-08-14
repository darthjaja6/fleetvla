# Contributing to FleetVLA

FleetVLA welcomes scheduling algorithms, workload definitions, and focused
backend or endpoint adapters. Start by running the CPU path:

```bash
python3 -m venv .venv && \
  . .venv/bin/activate && \
  python -m pip install -e '.[dev]' && \
  python -m ruff check . && \
  python -m mypy --strict examples/my_scheduler.py && \
  python -m pytest && \
  fleetvla demo
```

The `dev` extra matches the default CPU CI job, including NumPy-backed adapter
tests. CI also runs the optional Torch integration tests on Python 3.12. To run
that same job locally, install the CPU wheel and rerun the suite:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest
```

For a scheduler, first follow [the scheduler guide](docs/schedulers.md). A pull
request should keep the scheduler independent of runtime and environment
internals, pass `fleetvla test-scheduler`, and attach or describe a reproducible
benchmark artifact. Cite the originating paper or implementation and preserve
its license and attribution. Results should report the full workload rather
than a context-free scalar score. Wall-clock serving evaluates synchronous
`schedule()` methods on a deadline-bound worker and falls back to EDF on failure;
keep them bounded, never sleep inside `schedule()`, and disclose measured
decision cost for nontrivial search algorithms.
Local scheduler artifacts embed one source file. Keep portable examples
self-contained with standard-library and FleetVLA imports; otherwise include an
environment manifest and state that artifact replay depends on it.

For runtime or adapter changes, include tests for ordering, time units, stale
generations, resets, disconnects, buffer bounds, and deterministic virtual-time
behavior where relevant. Remote protocol changes must preserve version and
session admission, monotonic observation sequences, and local fallback.
Physical endpoints must retain local watchdogs and
safety limits; the central scheduler is not a safety controller.

Keep changes reviewable and avoid unrelated refactors or new dependencies.
Public contract changes need a design discussion before implementation. During
the pre-alpha period, compatibility is documented release by release; once a
stable contract is declared, removals will require a deprecation period.

## Maintainer and decisions

FleetVLA is currently maintained by [@darthjaja6](https://github.com/darthjaja6),
who owns release and public-contract decisions. Open a GitHub issue for design
discussion; the maintainer records the decision there before an incompatible
contract change is merged. This ownership will be updated here if the
maintainer group changes.

Use the repository issue forms before substantial scheduler, backend, or
endpoint work. A scheduler contribution should also update the
[scheduler gallery](docs/scheduler-gallery.md). Integration pull requests must
state the tested upstream revision and distinguish fake/protocol, simulator,
and physical validation. See the [compatibility policy](docs/compatibility.md)
and [security policy](SECURITY.md).
