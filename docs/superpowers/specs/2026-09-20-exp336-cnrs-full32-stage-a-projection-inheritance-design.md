# EXP-336 — C_NRS_CORE Full-32 Stage-A Projection-Inheritance Court

Status: **DESIGN / PREREGISTRATION ONLY — IMPLEMENTATION AND EXECUTION UNAUTHORIZED**

## 1. Authority

EXP-335 is closed with the independently verified disposition:

`AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`

The pre-result EXP-335 transition lock permits exactly the next scope:

`C_NRS_CORE_FULL32_STAGE_A_DESIGN_ONLY`

It does not authorize implementation, execution, Stage B, Stage C, EXP-302, EXP-320, scaling, 30M, or 100M.

EXP-336 therefore freezes a causal Stage-A design and nothing more.

## 2. Scientific question

Does the **exact registered EXP-335 full-32 subspace-projection repair** rescue the previously selected C_NRS_CORE Stage-A root-0 failure when CONTROL, SHAM, and PROJECT all continue from the exact same sealed C_NRS checkpoint, optimizer state, and RNG state?

The intervention under test is the projector. The architecture, world population, optimizer state, RNG state, learning rate, update order, evaluation thresholds, and resident parameter count are held fixed.

## 3. Why the starting state is C_NRS, not A_FIXED

A_FIXED and C_NRS_CORE share the same 10M budget and broad V0.17 backbone geometry, but they are not byte-compatible scientific residents:

- A_FIXED is stateless-restart compute.
- C_NRS_CORE is recurrent.
- C_NRS_CORE has loop conditioning.
- A_FIXED capacity bottleneck is 981.
- C_NRS_CORE capacity bottleneck is 973.

Transplanting an EXP-335 A_FIXED checkpoint into C_NRS would therefore add a second intervention and destroy causal interpretability.

Instead EXP-336 starts from the exact C_NRS_CORE state already selected by EXP-319's frozen Stage-A learning-rate selection.

## 4. Frozen C_NRS parent

EXP-319 repair run:

- run: `35311529822`
- Stage-A selection artifact: `10533822077`
- selection ZIP SHA-256: `ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559`
- selection JSON SHA-256: `4da236ea06a76df94ab6aa3e8141c6855a97f427f0b8614fc380d5c71e88e34a`
- selection authority digest: `39e5e846dbe87f3d5b34e438f0c8864626caa7049148fac934f2c893032e623f`

The selected C_NRS record is:

- learning rate: `3e-4`
- step: `1024`
- answer-token accuracy: `0.7007042253521126`
- greedy exact match: `0.25`
- EOS correctness: `1.0`
- answer-only loss: `2.149582388769822`
- original initial answer-only loss: `164.55180455597355`
- invalid-output rate: `0`
- nonfinite events: `0`
- original EXP-319 Stage-A floor: **FAIL**

Exact selected checkpoint artifact:

- artifact ID: `10534076546`
- name: `exp319-stage-a-C_NRS_CORE-0.0003-c3`
- artifact ZIP SHA-256: `d88a9d85584476ec5226321479560a1255f0d015122827d4be75f67a6f337755`
- `checkpoint.pt` SHA-256: `bd58607a0e49bb32f45124689d11880a13040afa702bf0524cce115d947f650b`
- `receipt.json` SHA-256: `cea3fb3d0b94e5d7cbec59813cfa3a1611dec0883a14acd6b2d0e46a23db3e4d`
- `summary.json` SHA-256: `32131c058049a27c24ac215cc04b77d5e1432cbda41fda71f8724f6723dfea23`
- model-state digest: `01c2a0f16b3f84dfee1b6db749821cf09897de05c6ceb15e42274abcd5926a76`
- optimizer-state digest: `d18439109cc9e020153b33fa6c39d38d4d0e05f2262c6a9ee3598df208f1ddf9`
- RNG-state digest: `3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95`

All three EXP-336 arms must reconstruct exactly this state before any update.

Fresh initialization, learning-rate retuning, and A_FIXED-to-C_NRS checkpoint transplantation are forbidden.

## 5. C_NRS resident

The resident is exactly `build_nrs_core_10m`:

- class: `V017RecurrentLM`
- trainable parameters: exactly `10,000,000`
- capacity bottleneck: `973`
- capacity tail: `192`
- weight-tied recurrence: yes
- loop conditioning: yes
- latent state carries only within one forward recurrent computation
- no persistent state is carried across examples in EXP-336

EXP-336 is not an always-active-state experiment.

## 6. World population

Use exactly the EXP-319 Stage-A root-0 population:

- generator: `exp319-worlds-v1`
- source: `materialize_stage_a()`
- 4 families
- 8 worlds/family
- 32 worlds total
- indices `0..7`
- frozen order:
  1. iterative-grid-and-maze
  2. algorithmic-sequence-transform
  3. generator-heldout-abstract-transformation
  4. language-sequence-control
  then index `0..7`.

The content IDs and bytes must match the population used by EXP-335 full-32 Stage-A. Any mismatch invalidates the court.

## 7. Arms

Three arms:

1. `CONTROL_CNRS_FULL32`
2. `SHAM_MEASURE_CNRS_FULL32`
3. `SUBSPACE_PROJECT_CNRS_FULL32`

All three start from the exact same C_NRS checkpoint, optimizer state and RNG state.

### CONTROL

Train normally on the source world only.

### SHAM

At every source update:

- compute the source gradient;
- at the identical pre-update model/optimizer/RNG state, measure gradients for all other 31 worlds;
- source and all 31 target backward passes use the same current exposure effort from the frozen 1/2/4/8 cycle;
- preserve the raw source gradient;
- apply the raw source gradient exactly as CONTROL does.

CONTROL and SHAM must be state-identical at every immutable chunk boundary. A mismatch is a dedicated fail-closed reducer state.

### PROJECT

Measure the same source + 31 target gradients at the same pre-update state, then apply the exact EXP-335 registered intervention.

## 8. Inherited projector

No post-hoc modification is allowed.

For target gradients whose raw source-target dot is negative and whose norm squared is above `1e-24`, assemble `T` in frozen target-world order and compute:

`g' = g - T pinv(T^T T) T^T g`

Frozen numerical rules:

- Gram system: float64
- `pinv` relative tolerance: `1e-12`
- target norm-squared floor: `1e-24`
- target order: frozen 32-world order excluding the source
- gradient clipping: after intervention
- half-space/no-new-conflict candidate: forbidden in EXP-336

Post-hoc EXP-335 leakage findings remain diagnostics only.

## 9. Continuation schedule

Starting cumulative C_NRS training step: `1024`.

Continue for:

- 32 additional exposures/world
- 32 worlds
- 1024 source updates total
- 8 immutable chunks
- 4 exposures/world/chunk
- 128 source updates/chunk

Schedule for each exposure `0..31`:

- family in frozen family order
- world index `0..7`

Effort cycle by exposure:

`1, 2, 4, 8`

Final cumulative C_NRS step: `2048`.

Optimizer semantics are inherited from the sealed checkpoint:

- AdamW
- learning rate `3e-4`
- weight decay `0.01`
- gradient clip `1.0`
- answer-only loss
- unchanged byte tokenizer

No learning-rate search is allowed.

## 10. Evaluation

Every chunk boundary is diagnostic. Only the final boundary is decisional.

### Primary full-32 teacher-forced floor

Every one of the 32 worlds is evaluated at frozen teacher-forced effort `4` and must simultaneously satisfy:

- teacher-forced answer-token accuracy >= `0.99`
- teacher-forced full-answer exact >= `0.90`
- nonfinite events = `0`

### Aggregate EXP-319 Stage-A sanity floor

At the final boundary, each arm must also satisfy:

- greedy exact match >= `0.90`
- answer-token accuracy >= `0.99`
- EOS correctness >= `0.95`
- final answer-only loss / original EXP-319 initial answer-only loss <= `0.25`
- original initial loss = `164.55180455597355`
- invalid-output rate <= `0.01`
- nonfinite events = `0`
- gating effort = `4`
- max generation tokens = `96`

An arm passes EXP-336 only if **both** the full-32 teacher-forced floor and the aggregate Stage-A sanity floor pass.

This prevents a teacher-forced-only success from being promoted as Stage-A readiness while greedy output remains weak.

## 11. Reducer

Frozen reducer order:

1. `INVALID_CNRS_FULL32_STAGE_A`
2. `PARENT_AUTHORITY_MISMATCH`
3. `SHAM_CNRS_FULL32_MISMATCH`
4. `CNRS_FULL32_STAGE_A_ESTABLISHED_BOTH`
5. `CNRS_FULL32_STAGE_A_ESTABLISHED_CONTROL_ONLY_PROJECT_REGRESSION`
6. `CNRS_BASELINE_FAILURE_WITHOUT_PROJECTOR_TRIGGER`
7. `CNRS_FULL32_STAGE_A_RESCUED_BY_INHERITED_PROJECTOR_NO_REGRESSION`
8. `CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`

World-level vectors must include:

- CONTROL failed worlds
- PROJECT failed worlds
- rescued CONTROL failures
- PROJECT regressions

A projector trigger requires both:

- `projection_update_count > 0`
- `projected_target_count > 0`

The rescue disposition requires CONTROL fail, PROJECT pass, a real projector trigger, at least one rescued CONTROL failure, and zero PROJECT regressions.

## 12. Pre-result successor lock

Before any EXP-336 execution exists:

- invalid → validity/provenance review only
- parent mismatch → parent authority reconciliation only
- SHAM mismatch → measurement non-perturbation localization only
- both pass → Stage-B design/preregistration only
- CONTROL-only pass → Stage-B design/preregistration only on CONTROL foundation
- no-trigger baseline failure → new causal hypothesis/preregistration only
- inherited-projector rescue with no regression → Stage-B design/preregistration only under exact inherited repair
- otherwise not established → new causal hypothesis/preregistration only

No EXP-336 outcome directly authorizes Stage-B implementation.

## 13. Explicitly forbidden claims

EXP-336 cannot directly establish:

- Stage B IID capability
- Stage C heldout transfer
- persistent always-active state
- EXP-302 implementation
- EXP-320 implementation
- a need to scale
- 30M
- 100M

It is a 10M Stage-A causal foundation court only.

## 14. Implementation boundary

At this checkpoint:

- `exp336_implementation_authorized = false`
- no EXP-336 runtime exists
- no EXP-336 workflow exists
- no execution identity exists
- no scientific run exists

A separate operator transition is required after this preregistration is frozen and CI-green.
