# Security

Report vulnerabilities privately to the maintainers through the repository's
security advisory feature. Do not include robot credentials, network addresses,
or unsafe reproduction steps in a public issue.

Benchmark artifacts can embed a local scheduler's Python source. FleetVLA never
executes that source during replay unless the user passes
`--allow-embedded-scheduler`; inspect and authenticate the artifact first.
Checksums detect modification but do not establish who created a file.

FleetVLA is research infrastructure, not a certified safety system. Physical
deployments require independent local limits, watchdogs, fallbacks, emergency
stops, and platform-specific validation.
