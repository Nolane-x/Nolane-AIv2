# EXP-282 Real EV-E3 Confirmatory-Open Closure Design

## Purpose

Close the currently unexecuted EXP-282 Stage-A small-model confirmatory-open lane with one real, lineage-closed EV-E3 ceremony using the scientific code that is already merged on `main`, while preserving the frozen Stage-A V1 protocol and explicitly keeping post-freeze 100M challenge replication (EV-E4) and independent clean-room replication (EV-E5) separate.

This is the first subproject in the broader Stage-A closure program. It does **not** attempt to close EXP-277, EXP-279, EXP-286, EXP-289, or the later EXP-282 EV-E4/EV-E5 tiers in the same change.

## Authoritative baseline

- Repository: `Nolane-x/Nolane-AIv2`.
- Baseline `main` SHA at design start: `77383b0a9c2ce92ded66f07234891790652a037b`.
- Frozen Stage-A protocol: `protocols/stage_a_v1.json`.
- Canonical frozen Stage-A V1 SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Existing EXP-282 scientific modules and CLIs are the authority for EV-E3. The ceremony change must not alter their scientific behavior.
- Existing EXP-282 Analysis Court explicitly caps this tier at EV-E3 and leaves `100M post-freeze challenge replication` open for EV-E4 and independent clean-room replication open for EV-E5.

The ceremony workflow must recompute the current `source_tree_digest(ROOT)` and protocol byte digest at runtime. Any hard-coded digest in workflow metadata is only an expected identity check, never a substitute for recomputation.

## Scope boundary

### In scope

1. Add a real EXP-282 EV-E3 GitHub Actions ceremony orchestration that reuses the existing development, prep, seal, executor, and Analysis Court CLIs.
2. Freeze and prove exact scientific source identity before confirmatory observations are consumed.
3. Generate a non-tiny same-source-tree DEVELOPMENT planning baseline and checkpoint.
4. Freeze confirmatory sample size and ordered reserved IDs using the existing prep contract.
5. Produce a two-phase ceremony seal before confirmatory execution.
6. Execute exactly one lineage-closed confirmatory-open evidence run when the prep is ready.
7. Persist immutable raw/analysis/bundle artifacts plus full provenance without rerunning the scientific experiment.
8. Preserve negative outcomes, including infeasible sample size, `HOLD_UNSTABLE`, and `KILL_SUBSYSTEM`.
9. Make the decision fork to the next EXP-282 subproject explicit.

### Out of scope

- Modifying `protocols/stage_a_v1.json` or its digest file.
- Modifying EXP-282 scientific source under `src/` or scientific CLIs under `scripts/` during the real ceremony/persistence branch.
- Materializing post-freeze challenge randomness in EV-E3.
- Claiming EV-E4, EV-E5, a 100M integrated result, or general Stage-A completion.
- Ad hoc reruns to obtain a preferable scientific outcome.
- Selecting a new sample size after confirmatory observations have been consumed.
- Building the full 100M EXP-282 matched challenge substrate; that is the next subproject only after an EV-E3 `PROMOTE_TO_NEXT_STAGE` result.

## Existing machinery to reuse unchanged

The ceremony must call the already-merged interfaces rather than duplicating scientific logic:

1. `scripts/run_exp282_paired_dev.py`
   - produces `NLM-EXP-282-PAIRED-DEV-EVAL-V1`;
   - writes the functional-only paired checkpoint;
   - produces the Neural Arm Registry DEVELOPMENT state.

2. `scripts/prepare_exp282_confirmatory_open.py`
   - freezes the pilot-derived paired variance, MESI, power, familywise alpha, sample size, and confirmatory replicate reservation;
   - remains EV-E2 / UNVERIFIED;
   - may return `NOT_READY_VARIANCE_EXCEEDS_MAX_N`.

3. `scripts/seal_exp282_confirmatory_ceremony.py`
   - derives execution and reconstruction authorizations;
   - seals protocol, checkpoint, source-tree, analysis, sample-size, and reserved-ID lineage;
   - remains EV-E2 / UNVERIFIED and consumes no confirmatory observations.

4. `scripts/execute_exp282_confirmatory_ceremony.py`
   - validates the existing seal and current source-tree digest;
   - executes only the reserved confirmatory-open IDs;
   - writes immutable raw EV-E2, analysis EV-E3, and result-bundle artifacts;
   - preserves `challenge_materialized=false`.

The real ceremony workflow may add provenance validation around these interfaces, but must not recreate their statistical decision rule in shell or workflow code.

## Evidence semantics

### DEVELOPMENT planning baseline

The planning run is development evidence only. It may be generated in the same Actions job before sealing because it does not consume confirmatory-open observations.

Freeze the non-tiny geometry and training/evaluation configuration in the workflow before execution. The recommended V1 ceremony geometry is the existing non-tiny CLI default unless runtime evidence proves it cannot execute reliably:

- `d_model=64`;
- `hidden_size=48`;
- `target_parameters=500000` per matched arm;
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

The exact values used must be recorded in the DEVELOPMENT artifact/checkpoint and ceremony summary. Changing them after seeing confirmatory data is forbidden.

### Sample-size freeze

The existing prep contract remains authoritative:

- frozen absolute MESI: `+0.03` grounded-decision accuracy;
- power target: `0.90`;
- familywise alpha: `0.05`;
- paired design;
- confirmatory `n` within `[32, 128]`;
- pilot observations cannot become confirmatory observations.

If the recomputed unclamped required `n` exceeds 128, the authoritative result for this ceremony attempt is **not ready**. The workflow must stop before seal/confirmatory execution and persist the planning/prep evidence as a negative engineering/scientific outcome. It must not clamp to 128, alter the pilot, change geometry, or rerun with alternate settings under the same ceremony identity.

### EV-E3 confirmatory-open

A ready prep permits exactly one sealed execution over the frozen reserved replicate IDs. Existing semantics remain unchanged:

- confirmatory observations are derived from the frozen Stage-A protocol root and domain-separated RNG streams;
- `confirmatory_data_consumed=true` only after the executor runs;
- `challenge_materialized=false` for all EV-E3 raw/analysis/bundle artifacts;
- raw remains EV-E2 / UNVERIFIED;
- Analysis Court emits at most EV-E3 and exactly one of `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, or `KILL_SUBSYSTEM`.

The primary decision rule remains the already-implemented paired bootstrap court with the Brier and accounted-FLOP protected endpoints. Workflow code may assert the result is structurally valid but may not replace or reinterpret that court.

## Freeze and execution architecture

### Phase 0 — orchestration-only branch

Create a ceremony branch directly from an exact verified `main` SHA. The implementation PR for this subproject may add only:

- `.github/workflows/exp282-real-ev-e3-ceremony.yml`;
- `.github/workflows/exp282-persist-real-ev-e3-evidence.yml`;
- design/plan documentation as needed.

Before any scientific execution, Actions must prove there is no diff from the designated source baseline under:

- `src/`;
- `scripts/`;
- `protocols/stage_a_v1.json`;
- `protocols/stage_a_v1.sha256`.

If any of these paths drift, the real ceremony aborts.

### Phase 1 — build planning baseline

The ceremony job installs the frozen model environment, verifies the canonical protocol, runs the non-tiny EXP-282 DEVELOPMENT pair, and writes:

- development paired execution JSON;
- Neural Arm Registry JSON;
- functional-only checkpoint.

The job validates the generated artifact and records checkpoint SHA-256, source-tree digest, protocol digest, execution-contract digest, model/world geometry, and exact runner configuration.

### Phase 2 — freeze prep

Run `prepare_exp282_confirmatory_open.py` exactly once on the generated development artifacts.

If status is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, stage an immutable planning bundle with the negative status and stop successfully as a **non-executed scientific closure state**, not as infrastructure failure. No confirmatory observation may be materialized.

If status is `CONFIRMATORY_OPEN_PREPARED`, continue.

### Phase 3 — seal

Run `seal_exp282_confirmatory_ceremony.py` and validate:

- EV-E2 / UNVERIFIED;
- exact confirmatory `n` and ordered reserved IDs;
- no confirmatory data consumed;
- `seed_materialization_status=NOT_EXECUTED`;
- `challenge_materialized=false`;
- code/checkpoint/protocol/prep/reconstruction lineage closed;
- ceremony seal self-hash valid.

The seal is immutable after this point.

### Phase 4 — execute once

Run `execute_exp282_confirmatory_ceremony.py` with the exact seal, paired execution artifact, and checkpoint used above.

The workflow then invokes existing validators on the returned raw, analysis, and bundle artifacts and asserts only structural boundaries:

- raw: EV-E2 / UNVERIFIED, confirmatory data consumed, challenge false;
- analysis: EV-E3, valid frozen decision, challenge false;
- result bundle: exact seal/raw/analysis digest lineage;
- current source-tree digest still equals the seal;
- checkpoint file SHA still equals the seal.

The workflow must not contain a hard-coded expected scientific decision or expected effect size.

## Immutable evidence bundle

For a ready/executed ceremony, upload one Actions artifact containing at least:

- `development-planning-baseline.json`;
- `development-arm-registry.json`;
- paired checkpoint or a checkpoint artifact plus its exact SHA-256 and GitHub artifact identity;
- `confirmatory-prep.json`;
- `ceremony-seal.json`;
- `confirmatory-raw.json`;
- `confirmatory-analysis.json`;
- `ceremony-result.json`;
- `ceremony-summary.json`;
- `source-identity.json`;
- `SHA256SUMS`.

For a not-ready sample-size outcome, the artifact contains all available planning/prep/source identity material and a summary explicitly stating that confirmatory data was never consumed.

`ceremony-summary.json` is descriptive metadata only. It must copy validated fields from the scientific artifacts and must never become a second decision engine.

## Durable persistence

Use a separate persistence-only workflow patterned after the successful EXP-297 evidence flow.

The persistence workflow must be pinned to the exact completed ceremony run/artifact identity. It must:

1. fetch GitHub Actions run metadata and artifact metadata;
2. require the run to be completed successfully on the expected ceremony orchestration SHA;
3. verify artifact name, ID, digest, and workflow-run binding;
4. download only that exact artifact;
5. verify every `SHA256SUMS` entry;
6. rerun artifact validators only as reconstruction/validation, never scientific execution;
7. write `PROVENANCE.md` and GitHub run/artifact metadata;
8. commit only under `evidence/exp282/<date>-real-ev-e3-confirmatory-open/`;
9. prove again that `src/`, `scripts/`, and frozen protocol files are unchanged.

The persistence workflow must never invoke `run_exp282_paired_dev.py`, prep, seal, executor, or Analysis Court to create a replacement scientific result.

## Anti-selection and failure policy

- Infrastructure failure before any confirmatory inference may be rerun only under a documented infrastructure-failure condition.
- Once confirmatory inference has begun, the scientific outcome is retained.
- A valid `HOLD_UNSTABLE` or `KILL_SUBSYSTEM` cannot be discarded and rerun for preference.
- A valid `NOT_READY_VARIANCE_EXCEEDS_MAX_N` cannot be converted into readiness by changing the planning geometry under the same ceremony identity.
- Artifact persistence failure may be retried because it does not rerun scientific inference; the retry must bind to the exact original run/artifact.
- No output file may be overwritten in place.

## Decision fork after durable EV-E3 persistence

### `PROMOTE_TO_NEXT_STAGE`

EXP-282 becomes eligible for the next subproject: **EV-E4 100M post-freeze public-beacon challenge replication**. Eligibility is narrow and does not itself establish a 100M result.

The EV-E4 subproject must design and freeze a genuinely integrated 100M matched comparison before consuming its future public beacon. It must not silently relabel the current ~500k matched belief pair as a 100M integrated result.

### `HOLD_UNSTABLE`

Persist the EV-E3 result and stop EXP-282 promotion. Any future attempt requires a new explicitly designed protocol/replication rationale; no same-protocol outcome-shopping rerun is allowed.

### `KILL_SUBSYSTEM`

Persist the EV-E3 result and mark the explicit-belief subsystem hypothesis as failed at this frozen Stage-A scope. Do not run EV-E4 for that hypothesis.

### `NOT_READY_VARIANCE_EXCEEDS_MAX_N`

Persist the negative planning outcome. Confirmatory execution remains unopened. A later redesign must be treated as a new development/protocol decision, not a continuation that hides the failed planning state.

## CI boundary

Ordinary CI must continue to exercise only synthetic/test ceremony paths already present. The real ceremony workflow is isolated from normal `push`/`pull_request` CI and must never execute real confirmatory observations as a routine regression test.

Before the ceremony workflow is eligible to run, the orchestration-only branch/PR must pass the existing full CI matrix on its exact head. Since scientific `src/scripts/protocol` bytes are unchanged, any regression indicates orchestration or environment drift and blocks execution.

## Test and verification plan

Because this subproject intentionally changes no scientific production code, implementation verification focuses on orchestration correctness:

1. YAML/static review of both workflows.
2. Exact-head normal CI 3/3 GREEN before real ceremony.
3. Workflow preflight proving zero scientific-source diff.
4. Protocol verification and runtime source-tree digest capture.
5. Runtime validation of development/prep/seal artifacts before confirmatory execution.
6. Runtime validation of raw/analysis/bundle artifacts after execution.
7. SHA256 bundle verification before upload.
8. Persistence workflow re-verifies GitHub metadata and all internal hashes.
9. Evidence-only diff audit before merge.
10. Post-merge `main` CI 3/3 GREEN after durable evidence PR merge.

## Scientific non-claims

Even a successful EV-E3 `PROMOTE_TO_NEXT_STAGE` supports only the frozen EXP-282 small-model partial-observability comparison between explicit belief state and recurrent hidden state. It does not establish:

- a 100M integrated NLM advantage;
- general probabilistic reasoning;
- general memory or world-model superiority;
- completion of all Stage-A gates;
- EV-E4 or EV-E5 evidence;
- any EXP-277/279/286/289 result.

## Implementation files for the next phase

After this spec is approved, the implementation plan should be limited primarily to:

- `.github/workflows/exp282-real-ev-e3-ceremony.yml`;
- `.github/workflows/exp282-persist-real-ev-e3-evidence.yml`;
- focused workflow/provenance tests or validation helpers only if they can be added without altering existing scientific behavior;
- PR/evidence documentation.

Any discovered need to modify `src/nolane_ai/experiments/exp282_*`, `scripts/*exp282*`, or the frozen protocol is an architectural escalation. Stop the ceremony implementation and redesign rather than modifying scientific code after this closure design.