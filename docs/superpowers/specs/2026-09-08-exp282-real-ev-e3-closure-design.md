# EXP-282 Real EV-E3 Confirmatory-Open Closure Design

## Purpose

Close the currently unexecuted EXP-282 Stage-A small-model confirmatory-open lane with one real, lineage-closed EV-E3 ceremony using the scientific code already merged on `main`, while preserving the frozen Stage-A V1 protocol and keeping post-freeze 100M challenge replication (EV-E4) and independent clean-room replication (EV-E5) separate.

This is the first subproject in the broader Stage-A closure program. It does **not** close EXP-277, EXP-279, EXP-286, EXP-289, or the later EXP-282 EV-E4/EV-E5 tiers.

## Authoritative baseline

- Repository: `Nolane-x/Nolane-AIv2`.
- Baseline `main` SHA at design start: `77383b0a9c2ce92ded66f07234891790652a037b`.
- Frozen protocol: `protocols/stage_a_v1.json`.
- Canonical frozen Stage-A V1 SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Existing EXP-282 scientific modules and CLIs remain the sole scientific authority for EV-E3.
- The existing EXP-282 Analysis Court caps this tier at EV-E3 and explicitly leaves `100M post-freeze challenge replication` open for EV-E4 and clean-room replication open for EV-E5.

Every ceremony run must recompute `source_tree_digest(ROOT)` and the protocol byte digest. Hard-coded identities in workflow metadata are expected-value guards, never substitutes for runtime recomputation.

## Scope boundary

### In scope

1. Add real EXP-282 EV-E3 GitHub Actions orchestration around the already-merged DEVELOPMENT, prep, seal, executor, and Analysis Court interfaces.
2. Prove exact scientific source identity before confirmatory observations are consumed.
3. Generate one non-tiny same-source-tree DEVELOPMENT planning baseline and functional-only checkpoint.
4. Freeze sample size and exact ordered confirmatory replicate IDs with the existing prep contract.
5. Seal checkpoint/code/protocol/analysis/reconstruction lineage before confirmatory execution.
6. Execute one valid confirmatory-open attempt when and only when prep is ready.
7. Persist immutable raw/analysis/bundle evidence and GitHub provenance without rerunning the scientific experiment.
8. Preserve negative states: infeasible `n`, `HOLD_UNSTABLE`, and `KILL_SUBSYSTEM`.
9. Make the decision fork to EXP-282 EV-E4 explicit.

### Out of scope

- Any modification of `protocols/stage_a_v1.json` or its digest file.
- Any modification of EXP-282 scientific modules under `src/` or EXP-282 scientific CLIs under `scripts/` for this ceremony.
- Post-freeze challenge materialization in EV-E3.
- EV-E4, EV-E5, 100M-integrated, or all-Stage-A claims.
- Outcome-shopping reruns, sample-size changes after confirmatory consumption, or geometry changes under an already-frozen ceremony identity.
- Building the full 100M matched challenge substrate; that is a separate subproject reachable only after EV-E3 `PROMOTE_TO_NEXT_STAGE`.

## Existing machinery to reuse unchanged

The ceremony calls existing interfaces rather than reproducing scientific logic:

1. `scripts/run_exp282_paired_dev.py`
   - emits `NLM-EXP-282-PAIRED-DEV-EVAL-V1`;
   - writes a functional-only paired checkpoint;
   - produces EXP-282 Neural Arm Registry DEVELOPMENT state.

2. `scripts/prepare_exp282_confirmatory_open.py`
   - freezes paired pilot variance, MESI, power, familywise alpha, sample size, and confirmatory reservation;
   - remains EV-E2 / UNVERIFIED;
   - may return `NOT_READY_VARIANCE_EXCEEDS_MAX_N`.

3. `scripts/seal_exp282_confirmatory_ceremony.py`
   - derives execution/reconstruction authorizations and seals the complete pre-execution lineage;
   - remains EV-E2 / UNVERIFIED with no confirmatory observation consumed.

4. `scripts/execute_exp282_confirmatory_ceremony.py`
   - validates the seal and current source tree;
   - executes only frozen reserved IDs;
   - writes immutable raw EV-E2, analysis EV-E3, and ceremony bundle artifacts;
   - preserves `challenge_materialized=false`.

Workflow code may add provenance and fail-closed assertions around these interfaces, but may not reimplement or reinterpret the statistical court.

## Predeclared ceremony geometry

The real ceremony uses the existing non-tiny matched-pair defaults, with the planning evaluation count increased to the required 32 replicates:

- `d_model=64`;
- `hidden_size=48`;
- `target_parameters=500000` per arm;
- `train_replicates=16`;
- `eval_replicates=32`;
- `eval_start_replicate=50000`;
- `batch_size=8`;
- `timesteps=6`;
- `variables=4`;
- `visibility_rate=0.5`;
- `noise_std=0.35`;
- `lr=0.002`;
- `weight_decay=0.0`.

Resource feasibility may be rehearsed **before** the ceremony orchestration head is declared final. Once the real ceremony workflow head has passed exact-head CI and is designated as the ceremony identity, these values are immutable. Any later geometry/configuration change requires a new orchestration commit and therefore a new ceremony identity before any valid real attempt is made. It may not be used to replace a valid scientific outcome from an earlier identity.

The exact configuration is recorded in the DEVELOPMENT artifact/checkpoint and copied into descriptive ceremony metadata.

## Sample-size freeze

The existing prep contract remains authoritative:

- absolute MESI: `+0.03` grounded-decision accuracy;
- power target: `0.90`;
- familywise alpha: `0.05`;
- paired design;
- confirmatory `n` within `[32, 128]`;
- DEVELOPMENT observations cannot become confirmatory observations.

If the recomputed unclamped required `n` exceeds 128, the authoritative state is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`. The workflow stops before seal/confirmatory execution and persists the planning/prep evidence. It must not clamp `n`, alter the pilot, change geometry, or rerun alternate settings under the same ceremony identity.

## EV-E3 evidence semantics

A ready prep permits one sealed execution over the exact frozen reserved replicate IDs. Existing semantics remain unchanged:

- confirmatory observations derive from the frozen Stage-A protocol root and domain-separated RNG streams;
- `confirmatory_data_consumed=true` only after executor invocation;
- `challenge_materialized=false` throughout EV-E3 raw/analysis/bundle artifacts;
- raw remains EV-E2 / UNVERIFIED;
- Analysis Court emits at most EV-E3 and exactly one of `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, or `KILL_SUBSYSTEM`.

The already-implemented paired bootstrap, Brier guard, accounted-FLOP guard, and decision ordering remain authoritative. Workflow assertions verify structure and lineage only; they do not hard-code an expected effect or decision.

## Freeze and execution architecture

### Phase 0 — orchestration-only branch

Create the ceremony branch from an exact verified `main` SHA. The implementation PR may add only:

- `.github/workflows/exp282-real-ev-e3-ceremony.yml`;
- `.github/workflows/exp282-persist-real-ev-e3-evidence.yml`;
- focused tests that inspect workflow/provenance contracts without changing scientific behavior;
- design/plan/PR documentation.

Before scientific execution, Actions proves zero diff from the designated scientific baseline under:

- `src/`;
- `scripts/`;
- `protocols/stage_a_v1.json`;
- `protocols/stage_a_v1.sha256`.

Any drift aborts the ceremony.

### Phase 1 — DEVELOPMENT planning baseline

The ceremony job installs the frozen model environment, verifies the canonical protocol, runs the predeclared non-tiny EXP-282 DEVELOPMENT pair, and writes:

- paired execution JSON;
- Neural Arm Registry JSON;
- functional-only checkpoint.

It validates the outputs and records checkpoint SHA-256, source-tree digest, protocol digest, execution-contract digest, model/world geometry, and exact runner configuration.

### Phase 2 — prep freeze

Run `prepare_exp282_confirmatory_open.py` once on the generated DEVELOPMENT artifacts.

- `NOT_READY_VARIANCE_EXCEEDS_MAX_N`: package a non-executed negative planning bundle and stop before any confirmatory inference.
- `CONFIRMATORY_OPEN_PREPARED`: continue.
- Any other status: fail closed.

### Phase 3 — seal

Run `seal_exp282_confirmatory_ceremony.py` and validate:

- EV-E2 / UNVERIFIED;
- exact `n` and ordered reserved IDs;
- `confirmatory_data_consumed=false`;
- `seed_materialization_status=NOT_EXECUTED`;
- `challenge_materialized=false`;
- checkpoint/protocol/prep/reconstruction/code lineage closed;
- self-hash valid.

The seal is immutable after this point.

### Phase 4 — execute once

Run `execute_exp282_confirmatory_ceremony.py` with the exact seal, paired execution artifact, and checkpoint.

After execution, call the existing validators and assert only the frozen boundaries:

- raw: EV-E2 / UNVERIFIED, confirmatory data consumed, challenge false;
- analysis: EV-E3, valid frozen decision, challenge false;
- bundle: exact seal/raw/analysis digest binding;
- current source-tree digest equals the seal;
- checkpoint SHA equals the seal.

No workflow assertion may require a particular scientific decision or effect size.

## First-valid-attempt authority

To remove post-result run selection, the ceremony identity is the exact final orchestration SHA plus its frozen scientific source identity.

For that identity:

1. The **first valid attempt** is authoritative.
2. A valid attempt includes `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, or `KILL_SUBSYSTEM`; all are retainable outcomes.
3. A later run on the same identity cannot replace a valid earlier result.
4. Rerun is allowed only when the earlier attempt failed for a documented infrastructure reason **before confirmatory inference began**.
5. If infrastructure fails after confirmatory inference begins, the partial/failure state must be preserved and reviewed before any new ceremony identity is created.

The persistence workflow must pin the exact first valid run/artifact, not choose among multiple completed artifacts after inspecting outcomes.

## Immutable evidence bundle

For an executed ceremony, upload one Actions artifact containing at least:

- `development-planning-baseline.json`;
- `development-arm-registry.json`;
- paired functional checkpoint (or separately bound exact checkpoint artifact identity plus SHA-256);
- `confirmatory-prep.json`;
- `ceremony-seal.json`;
- `confirmatory-raw.json`;
- `confirmatory-analysis.json`;
- `ceremony-result.json`;
- `ceremony-summary.json`;
- `source-identity.json`;
- `SHA256SUMS`.

For `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, persist all available DEVELOPMENT/prep/source identity material and a summary explicitly recording that confirmatory inference never began.

`ceremony-summary.json` is descriptive metadata copied from validated artifacts, never a second decision engine.

## Durable persistence

Use a separate persistence-only workflow patterned after EXP-297. It initially contains no outcome-specific scientific expectations. After the first valid ceremony run exists, an orchestration-only commit may pin its exact run ID, artifact ID, artifact digest, and ceremony SHA. That pinning commit must not touch `src/`, `scripts/`, or frozen protocol files.

Persistence must:

1. fetch GitHub run/artifact metadata;
2. require the pinned run to be completed on the expected ceremony SHA;
3. verify artifact name, ID, digest, and run binding;
4. download only that artifact;
5. verify every `SHA256SUMS` entry;
6. invoke existing validators only for reconstruction/validation, never to generate replacement science;
7. persist `PROVENANCE.md`, run metadata, and artifact metadata;
8. commit only under `evidence/exp282/<date>-real-ev-e3-confirmatory-open/`;
9. prove again that `src/`, `scripts/`, and frozen protocol files are unchanged.

The persistence workflow must never invoke DEVELOPMENT generation, prep, seal, executor, or Analysis Court builders to create a new result.

## Failure and anti-selection policy

- Pre-inference infrastructure failure may be retried only with documented evidence that confirmatory inference did not start.
- Once confirmatory inference starts, the resulting scientific state is retained.
- Valid `HOLD_UNSTABLE` and `KILL_SUBSYSTEM` are not rerunnable for preference.
- Valid `NOT_READY_VARIANCE_EXCEEDS_MAX_N` is not converted to readiness under the same identity.
- Persistence failure may be retried because it does not rerun science; every retry binds the exact original run/artifact.
- Outputs are append-only/no-overwrite.

## Decision fork after durable EV-E3 persistence

### `PROMOTE_TO_NEXT_STAGE`

EXP-282 becomes eligible for a new architectural subproject: **EV-E4 100M post-freeze public-beacon challenge replication**. This eligibility is narrow and does not itself establish a 100M result.

EV-E4 must design and freeze a genuinely integrated 100M matched comparison before consuming its future public beacon. The current ~500k matched belief pair cannot be relabeled as a 100M integrated result.

### `HOLD_UNSTABLE`

Persist and stop promotion. A future attempt requires a new explicit replication/protocol rationale; same-protocol outcome shopping is forbidden.

### `KILL_SUBSYSTEM`

Persist and mark the explicit-belief hypothesis failed at this frozen Stage-A scope. Do not proceed to EV-E4 for that hypothesis.

### `NOT_READY_VARIANCE_EXCEEDS_MAX_N`

Persist the failed planning state with confirmatory inference unopened. A later redesign is a new development/protocol decision, not a hidden continuation.

## CI boundary

Ordinary CI continues to exercise only synthetic/test paths. Real confirmatory execution is isolated from routine `push`/`pull_request` regression CI.

Before a ceremony head is designated authoritative, that exact head must pass the existing full CI matrix. Because scientific `src/scripts/protocol` bytes do not change, any regression blocks the ceremony.

## Verification plan

1. Focused tests/static validation for workflow/provenance contracts.
2. Exact-head normal CI 3/3 GREEN before real execution.
3. Zero scientific-source diff preflight.
4. Canonical protocol verification and runtime source-tree digest capture.
5. DEVELOPMENT/prep/seal validation before inference.
6. Raw/analysis/bundle validation after inference.
7. Full `SHA256SUMS` verification before upload.
8. First-valid-attempt metadata pinning.
9. Persistence metadata/hash/reconstruction validation.
10. Evidence-only diff audit before merge.
11. Post-merge `main` CI 3/3 GREEN after durable evidence integration.

## Scientific non-claims

Even EV-E3 `PROMOTE_TO_NEXT_STAGE` supports only the frozen EXP-282 small-model partial-observability comparison between explicit belief and recurrent hidden state. It does not establish:

- a 100M integrated NLM advantage;
- general probabilistic reasoning;
- general memory/world-model superiority;
- completion of all Stage-A gates;
- EV-E4 or EV-E5;
- any EXP-277/279/286/289 result.

## Implementation boundary

After approval, implementation planning should remain limited primarily to:

- `.github/workflows/exp282-real-ev-e3-ceremony.yml`;
- `.github/workflows/exp282-persist-real-ev-e3-evidence.yml`;
- focused workflow/provenance tests;
- PR/evidence documentation.

Any discovered need to modify `src/nolane_ai/experiments/exp282_*`, `scripts/*exp282*`, or the frozen protocol is an architectural escalation: stop and redesign rather than mutating scientific code during closure.