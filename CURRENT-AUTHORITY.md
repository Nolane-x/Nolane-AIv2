# Nolane-AIv2 V0.17 — Current Authority Snapshot

> **Repository-process snapshot only.** This file is maintained on the active V0.17 experiment branch for continuity. It is not part of the sealed EXP-335 scientific source and cannot alter a frozen court.

## Active lineage

The active research lineage is **V0.17 / A_FIXED 10M foundation recovery**.

The repository root on `main` still preserves the terminal V0.16.1 closure snapshot. That historical V0.16.1 state must not be mistaken for the active V0.17 experiment authority on this branch.

Current resident under test:

- arm: `A_FIXED`
- trainable parameters: exactly `10,000,000`
- scale change: unauthorized
- 30M: unauthorized
- 100M: unauthorized

## Active authoritative experiment

**EXP-335 — Full-32 A_FIXED Stage-A Foundation Re-entry**

Authoritative execution:

- PR: `#110`
- frozen preregistration head: `cfb7916077c8ffed97375a37ab2f9767d761a3cf`
- preregistration canonical digest: `db73212d75770ae3a3c0b2a3fb6f60544672970cc3ad112063b1b5bb232021cb`
- preregistration JSON SHA-256: `cc9f245fba352e9607400a6316db987253ccc41cbaf00dc8af9cd8cfe5952d88`
- implementation commit after separate operator authorization: `f80f765b8e5e79bed102dc138ce9575efac0ccf5`
- immutable scientific source: `d29e20e5af3467d16123e224462b37ef9a2ebf9d`
- source-tree digest: `4f39a06428f073a7ead40bc5d67e95d0681bf7a25ce5a060942189a88e33e864`
- scientific workflow SHA-256: `997eeb31733c7ee50f20f69001ba9fc75cba27ecca8f8bf94a520cff591d6e0e`
- sealed marker: `57fdf3a38b42b41951639a35dbb3c0d82055846b`
- execution digest: `24d4e97ad3c7026881a122cd309d671d9999f3f84b5dcc04a11c468c4409d520`
- authoritative run: `35445927525`

Only that run is scientifically authoritative. Duplicate run `35447261711` was cancelled and is explicitly non-authoritative.

## Current execution progress

At the latest repository-process update:

- `prepare`: **SUCCESS**
- `chunk0`: **SUCCESS**
- `chunk1`: **IN PROGRESS**
- `chunk2..chunk7`: not yet authoritative evidence
- `finalize`: not yet run
- final EXP-335 disposition: **NOT YET AVAILABLE**

Chunk 0 immutable evidence:

- artifact ID: `10586346937`
- artifact digest: `42db4c1f5389a9a566d02c212948df524fb0d1b74af87c0b778b4b2ce21461c9`
- chunk bundle digest: `ee63050f95f69c1ff493cafc01ec11b56ea5c18b5214e9d4b51ef6e5b6bed7a4`
- CONTROL/SHAM exact equality: true
- nonfinite events: 0 in all arms
- diagnostic worlds passing both thresholds: CONTROL `14/32`, SHAM `14/32`, PROJECT `15/32`

These chunk-level values are **diagnostic only**. They cannot authorize a successor or substitute for the final reducer.

## Frozen EXP-335 scientific geometry

- complete Stage-A root-0 population: 4 families × 8 worlds = 32
- 32 exposures/world
- world-local effort cycle `1,2,4,8`
- 8 immutable continuation chunks
- 128 source updates/chunk
- CONTROL / SHAM / SUBSPACE_PROJECT
- measured arms use all other 31 worlds as targets from identical pre-update state
- registered intervention:
  `g' = g - T pinv(T^T T) T^T g`
- float64 Gram
- pseudoinverse `rtol=1e-12`
- final per-world floors:
  - teacher-forced answer-token accuracy >= `0.99`
  - teacher-forced full-answer exact >= `0.90`
  - nonfinite events = `0`

No intermediate checkpoint can promote the experiment.

## Authority provenance

The design checkpoint intentionally recorded `exp335_implementation_authorized=false`.

A separate operator instruction existed before implementation and explicitly authorized implementation of the already-frozen EXP-335 design without changing its scientific geometry. Repository audit record:

`protocols/v017/exp335_operator_authorization_attestation_v1.json`

This attestation records pre-existing provenance; it is not retroactive authorization and does not modify the sealed execution.

## Pre-result successor lock

The allowed successor for every final reducer state was frozen before final-result visibility:

- machine-readable: `protocols/v017/exp335_post_result_transition_lock_v1.json`
- human-readable: `docs/superpowers/specs/2026-09-19-exp335-post-result-transition-lock.md`

Only three positive foundation states can open the next lane, and even then only as **C_NRS_CORE full-32 Stage-A design/preregistration**:

- `AFIXED_FULL32_FOUNDATION_REENTERED_BOTH`
- `AFIXED_FULL32_FOUNDATION_REENTERED_CONTROL_ONLY_PROJECT_REGRESSION`
- `AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`

All other final states keep C_NRS blocked and require validity review or a distinct new causal hypothesis.

## Post-hoc exploratory finding

Chunk-0 geometry analysis found that the registered projector nearly eliminates negative mass on selected conflicts, while most remaining negative mass is introduced on targets that had non-negative raw dots.

This is stored as **post-hoc, non-decisional** evidence:

`protocols/v017/exp335_chunk0_exploratory_geometry_v1.json`

A possible half-space / no-new-conflict projection is documented only as a future distinct hypothesis:

`docs/superpowers/specs/2026-09-19-post-exp335-no-new-conflict-halfspace-projection.md`

It must not be substituted into EXP-335.

## Contingency engineering branch

Draft PR `#111` is an execution-performance contingency, not authoritative science.

It preserves the registered scientific semantics while reducing redundant SHAM/projector compute. It may be used only if the authoritative execution becomes operationally unusable and a separate repair execution is explicitly authorized and resealed.

It must never be substituted for successful authoritative evidence merely because it is faster.

## Universal authorization guards

The following remain **false**:

- EXP-302 implementation
- EXP-320 implementation
- scale authorization
- 30M authorization
- 100M authorization

Even after a future positive C_NRS Stage-A result, Stage B and Stage C remain mandatory before any always-active persistent-state design can be justified.

## What to do next

1. Preserve run `35445927525` as the only authoritative EXP-335 execution.
2. Continue validating its immutable chunk chain.
3. Do not change EXP-335 scientific geometry based on intermediate evidence.
4. When chunk7 + final reducer exist, independently verify artifact digest, final JSON SHA, canonical evidence digest, CONTROL/SHAM integrity, chunk parent chain, and reducer.
5. Apply exactly the pre-result successor transition lock.
