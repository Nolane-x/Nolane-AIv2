# NLM V0.16.1 — Current Authority Snapshot

This file is the shortest current-state entry point for the repository. Historical README sections and older design documents remain useful for lineage, but they do not override this snapshot or the terminal closure audit.

## Canonical state

- Base lineage before terminal closure: `main@bd8696a3263f1be2c624a3ee995586a8acc455a0`
- Frozen Stage-A protocol: `protocols/stage_a_v1.json`
- Frozen Stage-A SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- Current V0.16.1 experiment namespace: `EXP-277..EXP-300`
- Current-lineage experiment IDs with explicit authority disposition: `24/24`
- Current-lineage experiment IDs without an authority disposition: `0/24`

## Terminal program state

`V0161_PROGRAM_EXECUTION_CYCLE_CLOSED`

The current V0.16.1 lineage has no remaining authorized experiment execution.

The decisive end state is:

- EXP-277 — `SCIENTIFICALLY_CLOSED_KILL`
- EXP-279 — `DEVELOPMENT_CLOSED_NEGATIVE`
- EXP-282 — `SCIENTIFICALLY_CLOSED_KILL`
- EXP-286 — `SCIENTIFICALLY_CLOSED_PROMOTE`
- EXP-287 — `DEVELOPMENT_CLOSED_NEGATIVE`
- EXP-289 — `SCIENTIFICALLY_CLOSED_PROMOTE`
- EXP-290 — `DEVELOPMENT_CLOSED_NEGATIVE`
- EXP-297 — `SCIENTIFICALLY_CLOSED_PROMOTE`
- EXP-298 — `DEVELOPMENT_CLOSED_POSITIVE`
- EXP-299 — `DEVELOPMENT_CLOSED_NEGATIVE`

All other EXP-277..EXP-300 IDs are explicitly `DEPENDENCY_REVIEW_REQUIRED` or `BLOCKED_ON_PARENT` under the current lineage; they are not silently pending executions.

## Why EXP-300 is not run

EXP-299's frozen scaffold-removal court closed:

`NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`

with:

- established roots: `0/4`
- successor design authorized: `false`
- authorization scope: `NONE`
- EXP-300 execution authorized: `false`

Therefore:

`EXP-300 = BLOCKED_ON_PARENT`

Running EXP-300 anyway would violate the current V0.16.1 dependency contract.

## Scale decision

`DO_NOT_SCALE_CURRENT_V0161_LINEAGE`

The current lineage did not satisfy the V0.16 conjunction required to justify scaling. This is a scoped research-program decision, not a universal claim that future NLM hypotheses cannot work.

## What remains true

Positive scoped evidence remains positive where earned:

- EXP-286: oracle conflict-core value in its frozen scope;
- EXP-289: episode-local nogood value in its frozen scope;
- EXP-297: bounded Encoding Fidelity Court value;
- EXP-298: bounded controlled-domain cross-domain fidelity transfer at DEVELOPMENT evidence level.

Negative results remain negative where observed and are not tuned away.

## What is still unverified

The repository does not claim:

- integrated NLM superiority;
- successful full 100M lifelong architecture;
- broad open-domain semantic authority;
- positive lifelong lemma economy;
- reliable native scaffold-free fidelity;
- positive full-lifecycle VCPF;
- benefit at larger scales.

## Canonical closure documents

Read these in order for current authority:

1. `docs/superpowers/specs/2026-09-13-v0161-terminal-program-closure.md`
2. `docs/superpowers/specs/2026-09-13-v0161-program-closure-matrix-design.md`
3. `docs/superpowers/specs/2026-09-13-exp299-development-closure.md`
4. `docs/superpowers/specs/2026-09-13-exp298-development-closure.md`

Historical experiment/design documents are retained for reproducibility and failure archaeology.

## Future-work rule

Any attempt to revisit a closed/blocked line must begin as a **scientifically distinct new hypothesis/version** with a new preregistration and pre-data authority gate. It must not be presented as a repaired or retuned rerun of a failed frozen court.
