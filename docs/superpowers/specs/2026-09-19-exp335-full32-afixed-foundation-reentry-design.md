# EXP-335 — Full-32 A_FIXED Stage-A Foundation Re-entry

**Status:** `DESIGN / PREREGISTRATION ONLY / IMPLEMENTATION BLOCKED`  
**Parent:** EXP-334 `HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION`  
**Model:** `A_FIXED`, exactly `10,000,000` trainable parameters  
**Scale authorization:** `false`  
**EXP-320 implementation authorization:** `false`

## 1. Why EXP-335 exists

EXP-319 ended at `TRAINING_STACK_NOT_LEARNABLE` because the frozen A_FIXED Stage-A control did not reach the preregistered local learnability floor. The subsequent causal chain established that this was not a simple inability of the 10M model to fit the individual Stage-A worlds:

- EXP-325: all 32 worlds can fit independently.
- EXP-326/327: shared training introduces localized interference.
- EXP-329: direct cross-update damage exists.
- EXP-330 through EXP-333: conflict-gradient projection rescues all observed pair failures across the complete 28-pair iterative lattice with no registered pair regression.
- EXP-334: simultaneous conflict-subspace projection rescues all registered higher-order iterative failures, including the full 8-world octet, with no registered higher-order regression.

EXP-334 does **not** establish the original four-family, 32-world Stage-A foundation. EXP-335 asks the next control-first question:

> From the immutable A_FIXED step-2048 reconstruction, can a balanced 32-world / four-family continuation make all 32 Stage-A worlds simultaneously satisfy the frozen teacher-forced foundation floor, either under the corrected schedule alone or under the already-validated conflict-subspace intervention generalized to all other worlds?

This experiment is a re-entry court for the A_FIXED supervised foundation only. It is not an NRS, persistent-state, or scaling experiment.

## 2. Immutable authority

EXP-335 is bound to:

- EXP-334 run `35438656413`
- EXP-334 artifact `10583626513`
- EXP-334 ZIP SHA-256 `8339fd376a23b7daeb8f45087eff634893557f37c47a117f308c7f54692b8ee8`
- EXP-334 final JSON SHA-256 `a393bf8fc237e2f2a357666059011e468ba7de06db3f3c62ecc3090f2aa25354`
- EXP-334 evidence digest `4cc1b2fa67610dc0614fcd6e03ac0aa122f5aa2c7f680dce01e2a29382105a05`
- immutable reconstruction artifact `10547681681`
- reconstruction checkpoint SHA-256 `4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5`
- reconstruction receipt SHA-256 `617ba665149a366d00782df5c4457a6a3d01081fe847aa144465377ee164bba3`

EXP-301 `KILL_H_RD_01` and EXP-319 `TRAINING_STACK_NOT_LEARNABLE` remain historical facts. EXP-335 cannot reinterpret either result.

## 3. Frozen population

The population is the complete Stage-A root-0 set:

1. `iterative-grid-and-maze`, indices `0..7`
2. `algorithmic-sequence-transform`, indices `0..7`
3. `generator-heldout-abstract-transformation`, indices `0..7`
4. `language-sequence-control`, indices `0..7`

Total: exactly `32` worlds.

The canonical world identifier is `<family>:<index>`.

No world may be dropped, replaced, reweighted, or selected after result visibility.

## 4. Frozen training geometry

Every arm starts from the same immutable A_FIXED step-2048 reconstruction.

Frozen optimizer/runtime controls:

- AdamW
- learning rate `5e-5`
- weight decay `0.01`
- global gradient clip `1.0`
- inherited optimizer state
- inherited RNG state
- no architecture change
- no tokenizer change
- no objective change
- no scale change

Each world receives exactly `32` additional exposures.

World-local effort is:

`1, 2, 4, 8, 1, 2, 4, 8, ...`

using exposure index `j mod 4`.

Within each exposure round, updates occur in frozen family order and then world index `0..7` inside each family.

Therefore each arm receives exactly `1024` authoritative source updates.

Diagnostic checkpoints are after world exposures `8, 16, 24, 32`. Only exposure `32` is primary.

## 5. Registered arms

### CONTROL_FULL32

Ordinary balanced training under the frozen schedule. No extra target-gradient measurement.

### SHAM_MEASURE_FULL32

For every source update:

1. capture the exact pre-update state and RNG;
2. measure the source gradient;
3. from that identical pre-update state, measure gradients for **all other 31 registered worlds**;
4. restore the source post-forward RNG state;
5. apply the untouched source gradient;
6. perform the ordinary global clip and AdamW step.

SHAM exists only to prove that the measurement path is non-perturbing.

### SUBSPACE_PROJECT_FULL32

Uses the same measurements as SHAM.

For source gradient `g`, every target gradient `t_i` satisfying:

- `g · t_i < 0`
- `||t_i||² > 1e-24`

enters the simultaneous conflict subspace.

Let `T` contain the selected target gradients conceptually as columns. The intervention is exactly:

`g' = g - T pinv(T^T T) T^T g`

with:

- Gram accumulation in float64
- pseudoinverse tolerance `rtol = 1e-12`
- ordinary clip `1.0` after projection
- ordinary AdamW update afterward

This is the direct all-world extension of the EXP-334 rule. No target sampling or post-result target selection is permitted.

## 6. Why the court must be chunked

A full measured arm requires the source gradient plus 31 target-gradient measurements for each of 1024 source updates. A monolithic standard-runner job is therefore intentionally forbidden.

Implementation, if later authorized, must use exactly:

- `4` exposures/world per continuation chunk;
- `128` authoritative source updates/chunk;
- `8` chained chunks.

Every continuation receipt must bind:

- arm
- chunk index
- cumulative exposure/world
- cumulative source updates
- model digest
- optimizer digest
- RNG digest
- data-order digest
- parent artifact digest
- exact source-tree digest
- preregistration digest

A chain position may have only one accepted parent. Cross-arm state splicing and silent replacement artifacts are forbidden.

CONTROL and SHAM for the same chunk must be executed on the same hosted runner/process boundary and must prove exact state equality before continuation. This preserves the within-run equality standard established after EXP-327 replay demonstrated that historical cross-run model/optimizer bitwise identity is not portable.

## 7. Primary metrics and gate

At exposure `32`, every one of the 32 worlds must simultaneously satisfy:

- teacher-forced answer-token accuracy `>= 0.99`
- teacher-forced full-answer exact `>= 0.90`
- nonfinite events `= 0`

The court also records:

- per-world answer-only loss
- per-world greedy exact
- negative target counts
- projected target counts
- projection-update count
- raw source-target dots
- post-projection target dots
- gradient norm before clipping
- parameter-update norm ratio

Intermediate checkpoints are diagnostic only and cannot promote an arm.

## 8. SHAM integrity

SHAM must exactly equal CONTROL at each accepted chunk boundary and final state in:

- teacher-forced world metrics
- model-state digest
- optimizer-state digest
- RNG-state digest

Any mismatch returns `SHAM_FULL32_MISMATCH` before a capability interpretation.

## 9. Reducer and dispositions

The reducer is control-first.

Decision order:

1. `INVALID_FULL32_FOUNDATION_REENTRY`
2. `PARENT_AUTHORITY_MISMATCH`
3. `SHAM_FULL32_MISMATCH`
4. `AFIXED_FULL32_FOUNDATION_REENTERED_BOTH`
5. `AFIXED_FULL32_FOUNDATION_REENTERED_CONTROL_ONLY_PROJECT_REGRESSION`
6. `BASELINE_FULL32_FAILURE_WITHOUT_SUBSPACE_TRIGGER`
7. `AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`
8. `AFIXED_FULL32_FOUNDATION_REENTRY_NOT_ESTABLISHED`

Required reducer vectors:

- CONTROL failed worlds
- PROJECT failed worlds
- rescued CONTROL failures
- PROJECT world regressions
- projection-update count
- projected-target count

Interpretation:

- `AFIXED_FULL32_FOUNDATION_REENTERED_BOTH`: balanced CONTROL already passes and PROJECT also passes.
- `AFIXED_FULL32_FOUNDATION_REENTERED_CONTROL_ONLY_PROJECT_REGRESSION`: CONTROL passes but PROJECT regresses at least one world. The A_FIXED foundation is re-entered under CONTROL, but the all-world intervention is not safe.
- `AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`: CONTROL fails, projection engages, PROJECT passes all 32 worlds, and no CONTROL-passing world regresses.
- `BASELINE_FULL32_FAILURE_WITHOUT_SUBSPACE_TRIGGER`: CONTROL fails but the registered intervention never activates.
- `AFIXED_FULL32_FOUNDATION_REENTRY_NOT_ESTABLISHED`: the full-32 A_FIXED foundation still does not satisfy the frozen floor.

## 10. Successor boundary

A positive A_FIXED re-entry result can authorize **design/preregistration only** for a future C_NRS_CORE full-32 Stage-A re-entry under the corresponding repaired foundation.

It does not authorize C_NRS implementation in this EXP-335 design phase.

Even after a future C_NRS Stage-A success:

- Stage B IID capability remains required;
- Stage C generator-heldout transfer remains required;
- EXP-320 always-active persistent-state design remains blocked until those gates are satisfied.

No outcome from EXP-335 authorizes:

- EXP-302 implementation
- EXP-320 implementation
- model scaling
- 30M
- 100M

## 11. Anti-rescue rules

After EXP-335 implementation is frozen and scientific execution begins, the following are forbidden:

- changing any of the 32 worlds;
- changing family/world order;
- changing exposure count;
- changing effort cycle;
- changing learning rate;
- changing the target norm floor;
- changing pseudoinverse tolerance;
- sampling only “difficult” targets;
- dropping target gradients to reduce runtime;
- relaxing the final per-world floors;
- selecting an earlier checkpoint because exposure 32 fails;
- dropping a failing world or family;
- changing the continuation chunk geometry based on partial scores;
- interpreting token-level improvement as foundation success when the primary gate fails.

A negative result may motivate a new preregistered causal hypothesis only.

## 12. Current authorization

At creation of this document:

- EXP-334 is positive and closed at `HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION`;
- EXP-335 implementation is **not authorized**;
- EXP-302 implementation is false;
- EXP-320 implementation is false;
- scaling is false;
- 30M is false;
- 100M is false.

The only authorized repository transition is this EXP-335 design/preregistration record.

## 13. Preregistration identity

Canonical preregistration digest:

`db73212d75770ae3a3c0b2a3fb6f60544672970cc3ad112063b1b5bb232021cb`

Exact preregistration JSON file SHA-256:

`cc9f245fba352e9607400a6316db987253ccc41cbaf00dc8af9cd8cfe5952d88`
