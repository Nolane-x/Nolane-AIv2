# EXP-277 Confirmatory Gate A/B Design

## Status and scope

This design extends the merged EXP-277 DEVELOPMENT substrate on `main@3e9cc7e3984febeac14f09d02e8a224e8deaaa95` into a frozen two-phase confirmatory program for `H-CBRF-01`.

The canonical Stage-A V1 protocol remains authoritative and MUST NOT be modified. EXP-277 remains:

- arms: `arcs_branch` versus `oracle_cbrf`
- question: whether oracle constraint/factor representation creates material headroom over ARCS at matched compute
- primary endpoint: `verified_utility_per_accounted_flop`, higher is better
- MESI: relative gain `+0.10`
- protected solution endpoint: `oracle_cbrf >= arcs_branch - 0.005`
- wall-energy endpoint: report-only at Stage-A and never omitted from the evidence ledger
- paired confirmatory sample-size bounds: `32 <= n <= 128`
- power target: `0.90`
- analysis method: paired effect with bootstrap confidence interval; report mean, median, sign consistency and divergence rate
- multiplicity family: `CBRF_HEADROOM`
- challenge lane: `POST_FREEZE_CHALLENGE`
- challenge seed rule: future public beacon only after protocol/code/config/evaluator/analysis and trained checkpoint identity are frozen.

A positive EXP-277 result does **not** validate a learned structure compiler, open-domain CBRF, or the integrated NLM architecture. It supports only the narrower claim that ground-truth structural information exposes material matched-cost headroom on the frozen synthetic structure-dense challenge family.

## Architectural choice

Use a dedicated two-phase immutable ceremony, modeled after the successful EXP-297 Gate A/Gate B pattern but with a pre-beacon trained-checkpoint freeze.

Gate A performs all training, DEVELOPMENT pilot evaluation, sample-size planning, checkpoint materialization, challenge-generator freeze, evaluator freeze, analysis freeze, reconstruction authorization and ceremony sealing **before any scientific beacon is selected or consumed**.

Gate B then imports a public beacon whose publication timestamp is strictly later than both:

1. the exact Gate A freeze commit timestamp; and
2. the Gate A trained-checkpoint seal timestamp.

Gate B derives challenge seeds deterministically, materializes hidden challenge worlds, loads the already-frozen trained arms, performs no optimizer step or parameter update, executes paired evaluation, reconstructs raw evidence, and applies the already-frozen analysis.

This design does not generalize all Stage-A experiments into one abstraction. EXP-277 receives dedicated modules because its relative primary endpoint, oracle-information separation, matched compute accounting and challenge geometry differ materially from EXP-282/297.

## DEVELOPMENT source and fixed pre-beacon geometry

Gate A consumes only `NLM-EXP-277-PAIRED-DEV-EVAL-V1` DEVELOPMENT evidence that validates with `validate_exp277_paired_development` and remains `EV-E2 / UNVERIFIED`.

The initial non-tiny Gate A geometry is frozen from the existing CLI defaults:

- `d_model=64`
- `hidden_size=48`
- `target_parameters=500000`
- `train_replicates=16`
- DEVELOPMENT pilot `eval_replicates=32`
- DEVELOPMENT pilot `eval_start_replicate=10000`
- `batch_size=8`
- `timesteps=4`
- `variables=6`
- `constraints=3`
- `noise_std=0.05`
- optimizer `AdamW`
- `lr=0.002`
- `weight_decay=0.0`

Changing this geometry after the first armed scientific ceremony begins creates a new ceremony identity and cannot be treated as a rerun of the original attempt.

Gate A must verify from the DEVELOPMENT artifact that:

- initial functional arm digests match
- total and functional parameter counts match
- oracle information is delivered only to `oracle_cbrf`
- paired arms share identical world lineage
- compute-budget court closes
- training uses only the DEVELOPMENT `augmentation` stream
- pilot evaluation uses only the DEVELOPMENT `evaluation` stream
- training and pilot replicate IDs are disjoint
- pilot has at least 32 paired replicates
- final functional-state digests are present for both arms
- protocol digest and source-tree digest are canonical and current.

Pilot observations are planning evidence only. They are never relabeled as confirmatory observations.

## Pre-beacon trained checkpoint

EXP-277 differs from EXP-297 because the candidate arms contain learned parameters. Gate A therefore freezes a trained functional checkpoint before the beacon.

The checkpoint builder deterministically reconstructs the matched pair from the frozen model-init seed and geometry, replays exactly the frozen 16 DEVELOPMENT training replicates, and verifies:

- training batch digests match the DEVELOPMENT artifact
- each stored per-arm training loss matches the reconstructed scalar with `rel_tol=0` and `abs_tol=1e-12`
- final functional-state digests exactly match the DEVELOPMENT artifact
- parameter/resource audit still closes after loading
- ARCS cannot receive oracle incidence
- optimizer state is not required by Gate B because no post-freeze training is permitted.

The checkpoint file may use a practical PyTorch serialization format, but its scientific identity is **not** the pickle/file bytes. Gate A computes canonical tensor-byte functional digests over ordered named trainable tensors and binds those digests into the seal. Gate B must load the checkpoint and recompute the canonical functional digests before any challenge inference.

If deterministic reconstruction cannot reproduce the frozen final-state digests, Gate A fails closed and no beacon may be consumed.

## Sample-size freeze

The DEVELOPMENT runner currently reports a per-replicate relative utility field using an epsilon denominator for engineering diagnostics. That epsilon-normalized quantity MUST NOT become the scientific confirmatory statistic.

Let each pilot replicate contain:

- `U_a(i)` = ARCS `verified_utility_per_accounted_flop`
- `U_o(i)` = oracle CBRF `verified_utility_per_accounted_flop`.

First compute the pilot baseline mean:

`mu_a = mean_i U_a(i)`.

If `mu_a <= 0` or is non-finite, Gate A returns `NOT_READY_PRIMARY_BASELINE_NONPOSITIVE`; no confirmatory n or beacon is permitted.

For planning only, define paired normalized pilot effects:

`d_i = (U_o(i) - U_a(i)) / mu_a`.

This preserves paired variance information without dividing by potentially-zero per-replicate baselines. The planning SD is the sample SD of `d_i`.

Use a one-sided normal approximation against MESI `0.10`, power `0.90`, and conservative familywise allocation across the primary and protected solution endpoint:

`alpha_per_endpoint = 0.05 / 2 = 0.025`.

The unclamped required n is rounded up. Gate A freezes:

`confirmatory_n = max(32, required_n)`

when `required_n <= 128`.

If required n exceeds 128, status is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`; no confirmatory replicate IDs are reserved and no beacon is consumed.

Reserved confirmatory replicate IDs are contiguous, ordered, deterministic and disjoint from DEVELOPMENT training and pilot IDs. Reservation begins at `max(all DEVELOPMENT replicate IDs) + 1`.

## Frozen primary statistic

The scientific primary effect is the paired-bootstrap ratio-of-means relative gain:

`R = mean(U_o) / mean(U_a) - 1`.

Gate B performs deterministic paired bootstrap resampling of confirmatory replicate rows with `10_000` samples and reports:

- observed `R`
- one-sided lower bound
- one-sided upper bound
- median bootstrap effect
- positive/zero/negative paired utility-difference counts
- paired sign consistency
- divergence/non-finite count.

No epsilon is permitted in the scientific denominator.

If observed confirmatory `mean(U_a) <= 0` or is non-finite, the primary endpoint is non-identifiable and the result is `HOLD_UNSTABLE` unless a separate frozen scientific kill condition is already satisfied.

For bootstrap resampling, **any** resample with `mean(U_a) <= 0` or non-finite marks the bootstrap distribution non-identifiable. The analysis must not drop, clip, replace or epsilon-correct such resamples. It records the invalid-resample count and returns `HOLD_UNSTABLE` unless a separate frozen scientific kill condition is already satisfied.

Undefined or infinite relative gains must never be converted into promotion.

## Protected solution endpoint

For each confirmatory replicate compute:

`S_i = verified_solution_rate_oracle - verified_solution_rate_arcs`.

Use deterministic paired bootstrap with the same frozen `10_000` samples and endpoint alpha `0.025`.

Promotion requires the one-sided lower confidence bound of the mean paired solution-rate difference to be `>= -0.005`.

A protected endpoint constitutes a decisive scientific failure only when its one-sided upper bound is `< -0.005`. Ambiguous overlap with the floor produces `HOLD_UNSTABLE`, not post-hoc threshold relaxation.

## Wall-energy evidence

The protocol requires `wall_energy_per_episode` to be report-only and impossible to hide.

GitHub-hosted runners are not assumed to provide a calibrated joule meter. Therefore every EXP-277 scientific artifact must include a wall-energy evidence object.

If no validated energy measurement backend was frozen before Gate A, the object must explicitly record:

- metric: `wall_energy_per_episode`
- `measurement_status=UNAVAILABLE_ON_HOSTED_RUNNER`
- `value_joules=null`
- hardware/environment identity when available
- reason the value is unavailable
- `used_for_decision=false`.

Wall clock, analytical FLOPs or CPU time must never be mislabeled as joules.

## Frozen result states and decision function

`INVALID_RUN` is reserved for provenance, resource-match, oracle-information-separation, reconstruction, checkpoint-identity or execution-integrity failures. Such failures are not scientific evidence for or against H-CBRF-01 and MUST NOT be relabeled as `KILL_SUBSYSTEM`.

`PROMOTE_TO_NEXT_STAGE` iff all of the following hold:

1. primary one-sided lower bootstrap bound `>= +0.10`;
2. protected solution-rate one-sided lower bootstrap bound `>= -0.005`;
3. resource-match and oracle-information-separation courts remain closed;
4. no invalid/divergent artifact condition applies.

`KILL_SUBSYSTEM` iff a scientifically valid run establishes either:

1. primary one-sided upper bootstrap bound `< +0.10`; or
2. protected solution-rate one-sided upper bootstrap bound `< -0.005`.

`HOLD_UNSTABLE` otherwise, including denominator instability or confidence intervals spanning decisive thresholds.

Practical equivalence therefore selects the simpler ARCS rival when the valid upper primary bound cannot reach the +10% MESI.

## Hidden post-freeze challenge generator

Gate A adds a dedicated EXP-277 challenge generator rather than relabeling DEVELOPMENT `evaluation` batches as confirmatory data.

The challenge-generator code is visible and frozen before the beacon; exact instances are unknowable until challenge seeds are derived after the seal.

The challenge family preserves the frozen structure-dense semantics and fixed outer geometry while deriving from the beacon seed:

- component-to-variable membership permutation
- binary component anchors
- normalized latent component codes
- anchor axis
- event component order
- surface and variable noise
- paired world identity.

Both arms receive identical `surface_events` and `variable_states`; only `oracle_cbrf` receives `oracle_incidence`. Targets and ground-truth structure are evaluator-only.

The challenge object must have a dedicated schema and scope such as `POST_FREEZE_CHALLENGE`. It must not accept the DEVELOPMENT root seed or the `augmentation/evaluation` stream API as an operator-selectable substitute for beacon-derived challenge material.

## Public beacon and exact frozen seed derivation

Gate A freezes a dedicated EXP-277 public-beacon receipt schema. Real Gate B receipts require:

- source
- beacon id
- publication timestamp UTC
- at least 256 bits of entropy
- evidence reference
- `test_only=false`
- `scientific_evidence_eligible=true`
- authenticity status `EXTERNAL_EVIDENCE_RECORDED` or `SOURCE_VERIFIED`
- canonical receipt digest.

The beacon must be strictly later than both the exact freeze commit timestamp and the runtime checkpoint-seal timestamp.

The frozen protocol seed rule is followed without adding post-hoc fields to the seed material. Canonical `beacon` means the validated beacon receipt digest. For each replicate/stream:

`SHA256(protocol_digest | beacon_receipt_digest | EXP-277 | stream | replicate)`.

The first 8 digest bytes are interpreted as a big-endian integer for deterministic challenge generation.

Freeze commit SHA and checkpoint seal digest are validated as independent authorization/provenance bindings; they are **not** silently added to the seed formula.

No Gate B CLI accepts an arbitrary `--seed` or equivalent operator-controlled challenge seed.

Synthetic beacons may be used in CI only when explicitly marked TEST-ONLY and scientifically ineligible.

## Gate A authorization and seal

Before any beacon is consumed, Gate A creates immutable authorization artifacts binding:

- canonical protocol digest
- exact source-tree digest
- DEVELOPMENT execution digest
- neural arm registry digest
- DEVELOPMENT final functional-state digests
- canonical checkpoint functional digests
- model/world/training geometry
- optimizer/training lineage
- pair-audit digest
- sample-size prep digest
- frozen analysis digest
- challenge-generator digest
- beacon/seed-derivation code digest
- executor digest
- reconstruction-court digest
- reserved confirmatory replicate IDs
- endpoint alpha allocation
- bootstrap count
- wall-energy evidence policy.

Gate A final seal state remains:

- evidence level `EV-E2`
- decision `UNVERIFIED`
- `confirmatory_data_consumed=false`
- `challenge_materialized=false`
- `seed_materialization_status=NOT_EXECUTED`
- `decision_rule_executed=false`
- trained checkpoint frozen and bound
- narrow readiness only: `FROZEN_MACHINERY_AND_CHECKPOINT_READY_FOR_FUTURE_BEACON_ONLY`.

The seal records both freeze commit time and checkpoint-seal creation time.

## Gate B raw executor

Gate B must:

1. validate the exact Gate A seal and current source-tree identity;
2. validate the public beacon against both freeze times;
3. load the pre-beacon checkpoint and reproduce canonical functional digests;
4. perform **zero** optimizer steps and parameter writes;
5. derive challenge seeds only through the frozen beacon function;
6. materialize each challenge replicate deterministically;
7. run both arms on the same paired challenge batch;
8. provide oracle incidence only to `oracle_cbrf`;
9. record exact analytical accounted-FLOP receipts;
10. record raw external metrics and challenge lineage;
11. emit an unanalyzed raw scientific artifact before analysis.

The raw validator must reject re-hashed tampering and independently reconstruct every deterministic challenge world and metric from lineage.

## Reconstruction court

For each raw confirmatory row the reconstruction court must reproduce or verify:

- reserved replicate ID
- beacon-derived challenge seed
- challenge batch digest
- world pairing
- evaluator-only targets
- oracle-incidence separation
- checkpoint functional digests
- ARCS and oracle raw predictions
- verified solution rate
- verified decision accuracy
- accounted FLOPs per episode
- verified utility per accounted FLOP
- paired utility difference
- paired solution-rate difference
- resource-match court status.

Aggregate values are never trusted when raw rows allow reconstruction.

Any reconstruction/provenance failure yields `INVALID_RUN`, not a scientific kill.

## First-valid-attempt ceremony discipline

The real scientific workflow follows the hardened pattern already proven in the repository:

1. dormant ceremony workflow and static contract tests are added to the experiment branch;
2. exact-head normal CI must be 3/3 GREEN while the workflow is unarmed;
3. the next scientific arming commit changes only the declared arm marker/orchestration metadata outside `src/` and `scripts/`;
4. the ceremony verifies exact source-tree identity and normal CI before science;
5. Gate A runs and seals the trained checkpoint before selecting a beacon;
6. an inference-start artifact is emitted before the first confirmatory forward pass;
7. the first valid scientific attempt on the exact armed identity is authoritative;
8. outcome-shopping reruns are forbidden;
9. infrastructure failure is rerunnable only under the frozen failure policy and only when the failure is proven to occur before first confirmatory inference;
10. persistence later pins that exact authoritative run/artifact digest and does not rerun science.

`PROMOTE`, `HOLD`, `KILL`, `INVALID_RUN` and valid `NOT_READY` are distinct outcomes and must not be conflated.

## CI boundary

Normal CI may exercise the complete Gate A/Gate B path using tiny geometry and a TEST-ONLY synthetic beacon. It must assert:

- TEST-ONLY outputs remain scientifically ineligible
- no TEST-ONLY result can promote H-CBRF-01
- arbitrary seed input is rejected
- beacon timestamps before either freeze boundary are rejected
- no parameter update occurs after Gate A checkpoint seal
- challenge truth/oracle incidence cannot leak to ARCS
- re-hashed tampering fails validation
- denominator-zero/instability cases do not promote
- protected-endpoint ambiguity produces HOLD rather than threshold relaxation
- provenance/resource/integrity faults produce `INVALID_RUN`, never `KILL_SUBSYSTEM`.

## TDD implementation decomposition

Implementation should proceed in small RED/GREEN tasks:

1. confirmatory prep and relative-effect sample-size freeze
2. deterministic trained-checkpoint builder/validator
3. post-freeze challenge generator
4. EXP-277 beacon receipt and seed derivation
5. execution authorization and Gate A seal
6. reconstruction court
7. frozen raw Gate B executor
8. frozen paired bootstrap analysis and decision rule
9. hardened Gate A/Gate B CLIs
10. TEST-ONLY end-to-end CI smoke
11. real dormant/armed ceremony workflow
12. persistence-only evidence workflow and final evidence integration.

Every major scientific module receives tamper tests before its GREEN implementation.

## Non-claims

Gate A is engineering/pre-beacon evidence only and cannot promote H-CBRF-01.

A future Gate B `PROMOTE_TO_NEXT_STAGE` would show only that the frozen oracle structural representation creates at least the preregistered matched-cost headroom on the frozen synthetic EXP-277 challenge family. It would **not** prove:

- a learned compiler can recover the oracle structure
- CBRF improves open-domain reasoning
- the integrated 100M architecture wins
- the effect scales to 1B+
- all Stage-A gates have passed.

A valid `KILL_SUBSYSTEM` outcome attacks the architectural justification for pursuing CBRF structure machinery under the frozen H-CBRF-01 claim and must not be softened after observing the result.
