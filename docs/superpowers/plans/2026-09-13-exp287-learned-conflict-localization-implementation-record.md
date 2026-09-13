# EXP-287 Learned Conflict Localization Court — Implementation Record

## Status

EXP-287 V1 was implemented and executed as a preregistered DEVELOPMENT / EV-E2 court. The experiment PR was closed without merge after a valid negative result. This record preserves the implementation lineage without importing the falsified runtime mechanism into `main`.

Base authority: `main@803fb474eca9cf57713f190e89ef17a113f335e5`.
Frozen Stage-A protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
Frozen EXP-287 geometry digest: `096a78bd6d7922c19943e1f3a249737fb5f3eff3325c5e7753d765e0b66e0dac`.

## Implemented work

### Task 1 — frozen geometry and matched tri-mode primitives

Implemented an EXP-287-specific synthetic conflict family and one canonical model per root with identical weights evaluated in three modes: `NULL_CORE_CONTROL`, `LEARNED_CORE_TOP2`, and `ORACLE_CORE_UPPER_BOUND`. The learned localizer used only model-visible state; the oracle mode was explicitly non-deployable.

### Task 2 — canonical training and same-weights evaluation

Each canonical root trained one model on augmentation-only data. Evaluation used disjoint held-out worlds after model freeze. Fixed training loss was rollback cross-entropy + verifier binary cross-entropy + localizer binary cross-entropy, all coefficient 1.0. No post-result calibration, class weighting, threshold sweep, early stopping, replacement seed, or sample escalation was permitted.

### Task 3 — fail-closed receipts and cross reducer

Root and cross receipts recomputed classifications from primitive metrics, rejected identity/boundary drift, used canonical JSON plus SHA-256 sidecars, and bound the frozen protocol, geometry, model state, root identity, and source-tree code digest.

### Task 4 — frozen CLI surface

Implemented fixed root and cross command surfaces with no user knobs for frozen geometry, roots, sample counts, top-k, loss weights, thresholds, or model dimensions. Publication was atomic/no-overwrite and validated before artifact creation.

### Task 5 — CI, one-shot DEVELOPMENT workflow, and freeze guard

The exact pre-data head `9b01353db39d22b0d7c799ee10c737f339467dce` passed dedicated EXP-287 CI, generic core 3.11, core 3.13, model-smoke, protocol verification, and freeze guard with zero prior EXP-287 DEVELOPMENT runs. The final pre-data review added direct `code_digest` binding to receipts and closed rerun/duplicate-run race conditions before release.

### Task 6 — marker-only release, execution, audit, closure

Release marker commit: `dd3789719a87f0bef47f09e17239e750c6d8bc1e`.
The commit changed exactly one file: `protocols/exp287-learned-conflict-localization-release.lock`.
Authoritative DEVELOPMENT run: `34733120820`.
Freeze guard independently observed exactly that run and no duplicates.

All four canonical roots completed successfully, root receipts and sidecars validated, and the cross reducer completed successfully.

## Result

All four roots reproduced positive oracle headroom and positive learned economic headroom while preserving verified-solution rate 1.0. Learned oracle-value capture was between approximately 0.719 and 0.758, above the frozen 0.50 economic threshold.

However, learned top-2 true-core precision was only 0.2207–0.2617 across roots, with cross-root mean `0.2392578125`, approximately the raw two-of-eight core prevalence `0.25`. Every root therefore failed the preregistered `top2_core_precision > 0.50` localization predicate.

Final cross decision: `LOCALIZATION_VALUE_NOT_ESTABLISHED`.

Authorization remained closed:

- `successor_design_authorized = false`
- `authorization_scope = NONE`
- `mechanism_successor_authorized = false`

EXP-288 therefore remains blocked under the current hypothesis.

## Evidence boundary

The sealed court remained DEVELOPMENT / EV-E2 only:

- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_labels_used_for_training = false`
- `evaluation_core_used_by_learned_mode = false`
- `oracle_mode_deployable = false`

## Artifact ledger

Cross artifact ID: `10310466510`.
Cross ZIP SHA-256: `83fa9f38b77abaa393f7be41a166d938009dc3e5b76fe182101d22e46ba7df92`.
Cross receipt SHA-256: `bdefb21515518dc48bbfc5560a790e2974c25e59a7cab212fbb0006acac88a8b`.
Cross canonical artifact digest: `de02a61119314d11aad1db86aa12fed7e80f3b208c498434f3cd829e91adcbb1`.

Root ZIP SHA-256:

- root 0: `788aacfd1cba5ec9940d4acda5433d34090c0e9b6f5c345bce946c9cfc084238`
- root 1: `94d7abfb2b2b5378395497b386081bab7a7a753f4e124c9a81a6478c201c1075`
- root 2: `9d29b0439761087b70e92b38e7bc69bc8da4ff4d90bb00d53d418878b6250824`
- root 3: `636e52db9f19ec68b266f94c1d9c141973a2650bad984a7289f7a8bd5ab7407d`

## Closure

PR #65 was closed without merge. The preserved experiment branch remains at marker head `dd3789719a87f0bef47f09e17239e750c6d8bc1e`; `main` remained `803fb474eca9cf57713f190e89ef17a113f335e5` at experiment closure. The negative result does not erase EXP-286 oracle conflict-core value; it localizes the current bottleneck to learned conflict localization under the frozen EXP-287 representation/objective.
