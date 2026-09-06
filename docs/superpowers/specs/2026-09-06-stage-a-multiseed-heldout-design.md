# Stage-A Multi-seed Held-out Development Design

## Goal

Measure seed-to-seed stability of the exact-16M Stage-A neural pilot without converting synthetic held-out development evidence into a confirmatory claim.

## Statistical unit and retention contract

- one model initialization/training/evaluation lineage is one seed outcome;
- RNG stream count is provenance, not replication;
- every requested seed index must appear exactly once and in order;
- failed seeds are emitted as explicit `FAILED` outcomes with seed lineage and error type; they may never be silently dropped or rerun into disappearance;
- if any requested seed fails, every aggregate mean/CI is structurally invalidated rather than recomputed over successful seeds;
- per-seed artifact/report digests are retained before any aggregate statistic;
- all six protected metric directions remain visible: belief accuracy, belief Brier reduction, conflict balanced accuracy, conflict Brier reduction, fidelity balanced accuracy, fidelity Brier reduction.

## Analysis contract

For each metric the development analysis retains mean, sample standard deviation, deterministic percentile-bootstrap 95% interval, and positive/zero/negative sign fractions. Bootstrap resampling uses a domain-separated SHA-256 counter generator (`sha256-counter-bootstrap-v1`) rather than global process RNG state.

These intervals are development diagnostics, not frozen confirmatory inference. Small-seed smoke runs test machinery only.

## Evidence boundary

The artifact is hard-capped at `EV-E2 / UNVERIFIED`. It cannot promote a neural claim, satisfy frozen Stage-A MESI, or substitute for post-freeze challenge/independent replication.
