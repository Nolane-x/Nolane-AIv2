# EXP-287 V1 DEVELOPMENT Closure Record

## Disposition

EXP-287 V1 is closed as a valid negative DEVELOPMENT / EV-E2 court.

Final cross decision: `LOCALIZATION_VALUE_NOT_ESTABLISHED`.

Authorization remains closed:

- `successor_design_authorized = false`
- `authorization_scope = NONE`
- `mechanism_successor_authorized = false`

EXP-288 therefore remains `BLOCKED_ON_PARENT` under the frozen EXP-287 hypothesis.

## Frozen identity

- base authority: `main@803fb474eca9cf57713f190e89ef17a113f335e5`
- exact pre-data head: `9b01353db39d22b0d7c799ee10c737f339467dce`
- release marker head: `dd3789719a87f0bef47f09e17239e750c6d8bc1e`
- Actions PR merge SHA: `a25e4269d7edb0a33675061148136a2d65049b72`
- authoritative DEVELOPMENT run: `34733120820`
- Stage-A protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- EXP-287 geometry digest: `096a78bd6d7922c19943e1f3a249737fb5f3eff3325c5e7753d765e0b66e0dac`
- source-tree code digest: `be413b357cabdbff5cd997fdbd50799e9ea0cd482b7418617dcdaa7dd7dcbd6b`

The release commit changed exactly one file, `protocols/exp287-learned-conflict-localization-release.lock`, and bound `pre_marker_head=9b01353db39d22b0d7c799ee10c737f339467dce`.

## Pre-data verification

Before the release marker existed, exact pre-data head `9b01353d...` passed:

- dedicated EXP-287 CI run `34732262149` — SUCCESS
- generic CI run `34732262156` — SUCCESS
  - core 3.11 — SUCCESS
  - core 3.13 — SUCCESS
  - model-smoke — SUCCESS
- freeze guard run `34732262154` — SUCCESS
  - no authoritative DEVELOPMENT run yet
  - no observed DEVELOPMENT run IDs
  - no duplicates

The pre-data review also closed two evidence-integrity defects before release: direct source-tree `code_digest` binding in sealed receipts, and rerun/duplicate-run race protection.

## One-shot execution

Release marker head `dd378971...` produced exactly one authoritative DEVELOPMENT run, `34733120820`.

Freeze guard run `34733120776` recorded:

- authoritative run `34733120820`
- observed run IDs `[34733120820]`
- duplicate run IDs `[]`

Execution completed successfully through preflight, all four canonical roots, receipt validation, artifact upload, cross reduction, cross validation, and final artifact upload.

## Four-root metrics

| root | control cost C0 | learned cost CL | oracle cost CO | capture | top-2 precision | learned solution rate | root decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2,320,512.0 | 1,880,883.75 | 1,740,384.0 | 0.7578125 | 0.251953125 | 1.0 | `LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED` |
| 1 | 2,320,512.0 | 1,892,969.75 | 1,740,384.0 | 0.7369791667 | 0.22265625 | 1.0 | `LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED` |
| 2 | 2,320,512.0 | 1,903,545.0 | 1,740,384.0 | 0.71875 | 0.220703125 | 1.0 | `LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED` |
| 3 | 2,320,512.0 | 1,887,682.125 | 1,740,384.0 | 0.74609375 | 0.26171875 | 1.0 | `LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED` |

Every root reproduced positive oracle headroom, positive learned economic headroom, capture above the frozen 0.50 economic threshold, and the protected solution-rate floor. Every root failed the frozen localization predicate `top2_core_precision > 0.50`.

Cross-root mean top-2 precision was `0.2392578125`, approximately raw two-of-eight core prevalence `0.25`. The frozen court therefore did not establish that the learned localizer was recovering the true conflict core, even though the learned execution mode reduced accounted cost versus null control.

This result does not prove that conflict-localization signal is impossible under every representation or objective. It closes only the frozen EXP-287 V1 hypothesis.

## Artifact ledger

Cross artifact:

- ID: `10310466510`
- name: `exp287-cross-a25e4269d7edb0a33675061148136a2d65049b72`
- ZIP SHA-256: `83fa9f38b77abaa393f7be41a166d938009dc3e5b76fe182101d22e46ba7df92`
- receipt SHA-256: `bdefb21515518dc48bbfc5560a790e2974c25e59a7cab212fbb0006acac88a8b`
- canonical artifact digest: `de02a61119314d11aad1db86aa12fed7e80f3b208c498434f3cd829e91adcbb1`

Root ZIP SHA-256:

- root 0: `788aacfd1cba5ec9940d4acda5433d34090c0e9b6f5c345bce946c9cfc084238`
- root 1: `94d7abfb2b2b5378395497b386081bab7a7a753f4e124c9a81a6478c201c1075`
- root 2: `9d29b0439761087b70e92b38e7bc69bc8da4ff4d90bb00d53d418878b6250824`
- root 3: `636e52db9f19ec68b266f94c1d9c141973a2650bad984a7289f7a8bd5ab7407d`

Independent audit verified receipt sidecars, canonical serialize/parse/serialize stability, cross artifact digest reconstruction, root artifact linkage, exact code/protocol/geometry identity, and run identity.

## Evidence boundary

The sealed cross receipt preserved:

- `evidence_level = EV-E2`
- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_labels_used_for_training = false`
- `evaluation_core_used_by_learned_mode = false`
- `oracle_mode_deployable = false`

## Post-release CI note

The marker-head dedicated and generic CI suites contain a prerelease-only sentinel requiring the release marker to be absent. Once the marker existed, that sentinel failed by construction. Dedicated CI reported `42 passed, 1 failed` with only that sentinel failing; generic core 3.11 likewise failed only that sentinel, while model-smoke passed. V1 was intentionally not modified after held-out result visibility merely to make a prerelease sentinel green.

## Repository closure

PR #65 was closed without merge. The experiment branch remains preserved at `dd3789719a87f0bef47f09e17239e750c6d8bc1e` for audit provenance. `main` remained `803fb474eca9cf57713f190e89ef17a113f335e5` at EXP-287 closure.

The correct next program authority is not EXP-288. The highest-priority surviving authorized seam is EXP-290, inherited from EXP-289's scoped EV-E3 promotion. EXP-298 remains independently authorized by EXP-297 and may be pursued separately.
