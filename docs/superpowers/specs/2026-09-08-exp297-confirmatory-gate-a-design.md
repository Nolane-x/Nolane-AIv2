# EXP-297 Confirmatory Gate A Design

## Status and scope

This design extends the merged EXP-297 DEVELOPMENT engineering baseline at `main@06fc1c530077095a01fb43711a9e3c73cc791326` with a pre-beacon confirmatory freeze and ceremony subsystem. Gate A MUST NOT consume confirmatory observations, materialize challenge instances, or derive challenge seeds from a public beacon. Gate A exists to freeze all code, configuration, evaluator logic, analysis logic, challenge-generation logic, lineage rules, and publication machinery needed for a later Gate B execution.

The frozen Stage-A protocol remains authoritative and MUST NOT be modified. In particular, EXP-297 remains:

- arms: `compile_only` versus `fidelity_court`
- primary endpoint: `semantic_fidelity_balanced_accuracy`, higher is better
- MESI: absolute gain `+0.10`
- protected endpoint: `wrong_formalization_authority_rate <= 0.05`
- protected endpoint: `faithful_formalization_rejection_rate <= 0.10`
- paired confirmatory sample-size bounds: `32 <= n <= 128`
- power target: `0.90`
- analysis method: `paired balanced-accuracy difference with Wilson/bootstrap safety intervals for wrong-authority rate`
- multiplicity family: `ENCODING_FIDELITY`
- challenge lane: `POST_FREEZE_CHALLENGE`
- challenge rule: future public beacon only after protocol/code/config/evaluator/analysis freeze.

Gate A artifacts remain `EV-E2 / UNVERIFIED`. They must explicitly state:

- `confirmatory_ready=false` until the pre-beacon ceremony seal is valid
- `confirmatory_data_consumed=false`
- `seed_materialization_status=NOT_EXECUTED`
- `challenge_materialized=false`
- `decision_rule_executed=false`
- `semantic_authority_promoted=false`

A valid Gate A seal may set a narrow readiness status such as `CONFIRMATORY_GATE_A_SEALED`, but it MUST NOT change evidence level, scientific decision, or semantic-authority status.

## Architectural choice

Use a two-phase immutable ceremony.

Gate A freezes machinery before any beacon exists. Gate B later imports a public-beacon receipt whose publication time is strictly after the Gate A freeze commit, derives challenge seeds deterministically, materializes challenge worlds, executes the frozen arms without tuning, reconstructs raw evidence, and runs the already-frozen analysis.

This design intentionally does not generalize every Stage-A experiment into one abstraction. EXP-282 remains a reference implementation, while EXP-297 receives dedicated modules because its semantics, trap families, witness court, and safety endpoints differ materially.

## Development pilot source and sample-size freeze

Confirmatory prep consumes only DEVELOPMENT evidence produced by `NLM-EXP-297-PAIRED-DEV-EVAL-V1`. The pilot artifact must:

- validate with `validate_exp297_execution`
- remain `EV-E2 / UNVERIFIED`
- contain at least 32 DEVELOPMENT replicates
- expose complete deterministic raw candidate rows
- keep evaluator truth outside arm-observable inputs
- bind protocol digest, code digest, matched-pair audit, and execution digest.

Pilot observations are used only for variance planning and can never be reclassified as confirmatory observations.

For each replicate, reconstruct the two arms' balanced accuracy from that replicate's 16 candidate rows and compute the paired effect:

`fidelity_court_BA - compile_only_BA`.

The confirmatory planning variance is the sample SD of those per-replicate paired effects. The frozen sample-size calculation uses a one-sided normal approximation against MESI `0.10`, power `0.90`, and familywise alpha `0.05`, with conservative Bonferroni allocation across the primary endpoint and two protected endpoints:

`alpha_per_endpoint = 0.05 / 3`.

The unclamped required n is rounded up, then confirmatory n is frozen to `max(32, required_n)` when the result is `<=128`. If required n exceeds 128, status is `NOT_READY_VARIANCE_EXCEEDS_MAX_N` and no confirmatory replicate IDs are reserved.

Reserved confirmatory replicate IDs must be contiguous, ordered, deterministic, and disjoint from every DEVELOPMENT replicate represented in the pilot artifact. The reservation starts at `max(development_replicate_ids) + 1`.

## Frozen analysis contract

The prep artifact freezes an analysis object containing exactly these scientific commitments:

- primary endpoint: `semantic_fidelity_balanced_accuracy`
- direction: higher
- paired contrast: `fidelity_court - compile_only`
- MESI: `0.10`
- power target: `0.90`
- familywise alpha: `0.05`
- endpoint alpha: `0.05 / 3`
- primary inference: one-sided lower bootstrap confidence bound
- wrong-authority guard: one-sided Wilson upper bound `<=0.05`
- faithful-rejection guard: one-sided Wilson upper bound `<=0.10`
- bootstrap sample count: `10_000`
- multiplicity family: `ENCODING_FIDELITY`
- candidate truth and trap family remain evaluator-only
- `court_inconclusive` grants no authority and counts as rejection for faithful-rejection accounting.

The frozen decision function for Gate B is:

- `PROMOTE_TO_NEXT_STAGE` only if the one-sided lower bootstrap bound of the paired BA gain is `>=0.10`, the one-sided Wilson upper bound for wrong-authority is `<=0.05`, and the one-sided Wilson upper bound for faithful-rejection is `<=0.10`.
- `KILL_SUBSYSTEM` if either protected safety guard fails, or if the one-sided upper bootstrap bound of the paired BA gain is `<0.10`.
- `HOLD_UNSTABLE` otherwise.

A promotion remains scoped to the frozen synthetic EXP-297 challenge and does not establish general natural-language semantic understanding.

## Hidden challenge generator

Gate A introduces a separate challenge generator rather than reusing the DEVELOPMENT generator as confirmatory data.

The challenge generator code is visible and frozen before the beacon, but exact instances remain unknowable because every replicate is parameterized by a post-freeze challenge seed.

The generator preserves the frozen eight semantic trap strata:

1. `faithful_equivalent`
2. `relation_shift`
3. `constraint_drop`
4. `constraint_strengthen`
5. `constraint_weaken`
6. `variable_binding_swap`
7. `domain_mapping_error`
8. `negation_or_relation_flip`

Each replicate contains one faithful and one wrong candidate per stratum, for 16 candidates total. Candidate order is deterministically shuffled from the challenge seed. Arm-visible candidate IDs are opaque. Constraint labels and `world_id` values are neutral and must not contain polarity, stratum, or trap taxonomy.

Unlike the DEVELOPMENT generator, the challenge generator varies hidden formal geometry from the seed, including domain size, relation offset, relation orientation, allowed-row ordering, neutral variable-name permutation, and transformation parameters. The source and all candidates remain compile-valid under the frozen `compile_valid` contract. Construction provenance must be cross-checked against `FidelityCourt` exact semantics for every generated case.

## Public-beacon receipt and seed derivation

Gate A MUST NOT fetch, choose, or materialize a beacon. It only freezes the receipt schema and seed derivation algorithm.

Gate B supplies a public-beacon receipt with:

- `source`
- `beacon_id`
- `published_at_utc`
- `entropy_hex`
- `evidence_reference`
- `receipt_digest`.

The Gate B ceremony must require the beacon publication timestamp to be strictly later than the Gate A freeze commit timestamp recorded in the seal. Gate A code validates syntax and canonical digest only; authenticity of the external public-beacon observation is a separately recorded ceremony evidence obligation unless a source-specific verifier is added before the Gate A freeze.

For each reserved replicate and each frozen stream, derive challenge seed material with:

`SHA256(protocol_digest | freeze_commit_sha | beacon_receipt_digest | EXP-297 | stream | replicate)`.

The challenge generator receives only the resulting deterministic integer seed. Operators cannot pass an arbitrary challenge seed through the CLI.

## Execution authorization

Before a beacon is consumed, create an `AUTHORIZED_NOT_EXECUTED` artifact that binds:

- canonical Stage-A protocol digest
- DEVELOPMENT pilot execution digest
- EXP-297 extended registry digest
- prep digest
- frozen-analysis digest
- sample-size-freeze digest
- challenge-generator code digest
- evaluator/reconstruction code digest
- execution code digest
- matched-arm pair audit digest
- reserved confirmatory replicate IDs
- expected candidate count per replicate
- max exact assignment ceiling
- frozen model geometry
- frozen `FidelityCourt` semantics.

The authorization artifact must contain no beacon value, no materialized challenge seed, and no challenge candidate.

## Reconstruction court

Gate A freezes a reconstruction authorization describing what Gate B validation must reproduce independently.

For each future raw challenge row, the validator must reconstruct from lineage:

- beacon-derived replicate seed
- challenge world and candidate order
- candidate opaque ID and canonical digest
- evaluator-only stratum and truth
- compile validity
- exact FidelityCourt receipt and witness
- fail-closed authority decision
- neural pair geometry and initialization
- analytical neural FLOPs
- semantic verification operations
- total accounted cost proxy
- raw confusion-matrix contributions
- per-replicate BA contrast
- global safety counts.

Re-hashing a modified artifact must not make forged data valid.

## Ceremony seal

The final Gate A artifact is a pre-beacon seal. It embeds or cryptographically binds the prep, execution authorization, reconstruction authorization, frozen analysis, challenge contract, and baseline DEVELOPMENT identities.

Required seal status:

- schema: `NLM-EXP-297-CONFIRMATORY-GATE-A-SEAL-V1`
- evidence level: `EV-E2`
- decision: `UNVERIFIED`
- status: `CONFIRMATORY_GATE_A_SEALED`
- `confirmatory_ready=true` only in the narrow sense that the frozen machinery is ready for a future-beacon execution
- `confirmatory_data_consumed=false`
- `seed_materialization_status=NOT_EXECUTED`
- `challenge_materialized=false`
- `decision_rule_executed=false`
- `semantic_authority_promoted=false`.

The seal records `freeze_commit_sha` and `freeze_commit_timestamp_utc`. Gate B must refuse a beacon whose publication timestamp is not strictly later.

Any production-code change affecting protocol interpretation, challenge generation, evaluator logic, court semantics, matched arms, reconstruction, analysis, or ceremony after the seal invalidates the seal and requires a new Gate A freeze.

## Gate B code frozen during Gate A

To avoid changing code after seeing a beacon, Gate A includes but does not execute the complete Gate B software path:

- beacon receipt validation
- deterministic challenge-seed derivation
- challenge-world generation
- confirmatory raw executor
- confirmatory raw validator
- frozen confirmatory analyzer
- confirmatory analysis validator
- two-phase ceremony runner/CLI.

CI may exercise those functions only with explicit synthetic test beacons whose timestamps and entropy are marked TEST-ONLY. Such test data is infrastructure verification and must never be emitted as scientific confirmatory evidence.

## CLI and publication safety

Add Gate A CLI commands that:

- verify canonical frozen protocol digest before any work
- refuse output overwrite
- validate every input artifact
- refuse DEVELOPMENT pilots with fewer than 32 replicates
- build prep, execution authorization, reconstruction authorization, and seal transactionally
- never accept beacon material during Gate A
- stage outputs and publish atomically
- emit only EV-E2 / UNVERIFIED readiness metadata.

Add a separate Gate B CLI that is present and testable but not invoked by normal CI. It requires a Gate A seal plus a beacon-receipt JSON file and refuses direct seed input.

## Tamper and leakage court

Tests must prove rejection of re-hashed tampering in at least these classes:

- pilot truth label, stratum, candidate order, candidate digest, witness, authority, or cost
- sample-size SD, required n, frozen n, or reserved IDs
- overlap between DEVELOPMENT and confirmatory replicate IDs
- primary metric, MESI, alpha allocation, bootstrap count, or safety thresholds
- challenge-generator digest or code lineage
- trap/polarity leakage into arm-visible challenge objects
- beacon digest, beacon timestamp, freeze commit SHA, or seed derivation
- arbitrary operator-supplied challenge seed
- `court_inconclusive -> court_accept`
- wrong-authority or faithful-rejection counts
- confirmatory/challenge flags
- decision-rule execution before raw challenge consumption
- semantic-authority promotion.

## CI and package version

Gate A adds focused tests for prep, challenge worlds, authorization, reconstruction, ceremony seal, synthetic-beacon Gate B machinery, analyzers, and CLI publication. Normal CI must execute only test/synthetic beacons and must assert that no scientific confirmatory artifact is produced from them.

Package target after Gate A integration: `0.16.0`.

## Non-claims

Gate A does not execute EXP-297 confirmatory data and cannot produce `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, or `KILL_SUBSYSTEM` as scientific outcomes. It only freezes and verifies the machinery required for a later future-beacon Gate B.

Even after a future Gate B promotion, the claim remains restricted to the frozen synthetic semantic-formalization challenge family. It does not validate unrestricted open-language formalization, general semantic understanding, or unconstrained formal authority.
