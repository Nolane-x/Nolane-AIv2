# EXP-331R — Parent Anchor Representation-Normalization Repair

The authoritative EXP-331 run `35420761503` completed with workflow success but reduced to `PARENT_PAIR_LATTICE_REPRODUCTION_MISMATCH`.

Forensic comparison of its immutable final artifact showed that every CONTROL arm matched the authoritative EXP-327 parent anchor in members, final token metrics, final full-answer exact metrics, model-state digest, optimizer-state digest, RNG-state digest, nonfinite count, and pass/fail status. The false mismatch arose before JSON serialization because EXP-331 compared dataclass-produced metric tuples against JSON-loaded metric lists.

EXP-331R is an implementation-only repair. It normalizes both sides of the two parent metric comparisons to tuples before exact equality. It does not alter the preregistration, model, data, optimizer, learning rate, gradient clipping, exposure budget, effort cycle, pair lattice, projection rule, reducer decision order, thresholds, authorization flags, or scientific workflow.

The repair adds a regression test whose runtime record uses tuple metrics and whose parent anchor uses JSON-style list metrics. The test requires the exact-state parent gate to accept equal values across those representation types.

The original EXP-331 evidence remains preserved as fail-closed evidence. EXP-331R must pass the repository CI, inherited EXP-301 contract, and EXP-331 contract before a new execution identity is sealed and the court is rerun.

EXP-302, EXP-320, scaling, 30M, and 100M remain unauthorized.
