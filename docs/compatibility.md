# Compatibility and releases

FleetVLA is pre-alpha. Artifact, public dataclass, scheduler, endpoint, and
backend contracts are versioned independently of workload names. The current
algorithm artifact format is version 1. System-track artifacts also use schema
version 1 but identify themselves with `artifact_kind: "system"`; their
checksums protect provenance and raw events, while their wall-clock runs are not
claimed to replay bit-for-bit. Readers reject unknown versions and modified
checksums rather than guessing. `fleetvla verify-artifact FILE` handles either
kind without executing an embedded scheduler; `fleetvla replay FILE` remains
algorithm-only. Development artifacts include an exact SHA-256 identity over
the installed FleetVLA Python source in addition to the package version.

Until 1.0, each release notes incompatible changes. Once a public contract is
declared stable, removals require a deprecation warning for at least one minor
release and a migration example. Versioned benchmark JSON files are immutable;
behavior changes create a new workload version instead of rewriting past
results.

CI covers Python 3.10 and 3.12 for the dependency-free core. Current LeRobot
0.6.2 integrations require Python 3.12 and are optional. A separate Python 3.12
CPU-Torch job exercises the LeRobot `[B,H,D]` batching adapter without making
Torch a core dependency. GPU model revisions,
simulator commits, robot firmware, and ROS 2 distributions must be pinned in
reported deployment artifacts even when FleetVLA itself is unchanged.
