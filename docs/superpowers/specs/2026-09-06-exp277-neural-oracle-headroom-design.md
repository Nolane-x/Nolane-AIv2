# EXP-277 Neural Oracle-Structure Headroom Design

## Purpose

Turn the frozen Stage-A EXP-277 contract into an executable small-neural development lane without changing `protocols/stage_a_v1.json` or claiming neural evidence prematurely.

EXP-277 asks whether ground-truth constraint/factor structure creates material matched-cost headroom over the V0.15 ARCS-style recurrent branch substrate. The frozen primary endpoint is `verified_utility_per_accounted_flop`, the MESI is a 10% relative gain, and the protected solution-rate floor is `oracle_cbrf >= arcs_branch - 0.005`.

## Scientific boundary

- Frozen Stage-A V1 protocol bytes and digest remain unchanged.
- This PR may create only DEVELOPMENT artifacts (`EV-E2 / UNVERIFIED`).
- No confirmatory-open replicate may be consumed.
- No post-freeze challenge randomness may be materialized.
- The development lane must not call a proxy implementation a confirmatory protocol arm unless parameter, information-lineage and compute accounting are explicit.
- A positive development result does not promote H-CBRF-01.

## Architecture

### 1. Matched experiment-local neural arms

Create two experiment-local neural arms under one exact target-parameter envelope:

- `ARCSBranchArm`: consumes the same underlying problem information through a surface/event representation and recurrent branch/deliberation substrate. It receives no ground-truth factor-incidence tensor.
- `OracleCBRFArm`: consumes the same problem plus the ground-truth factor-incidence tensor through an explicit constraint/factor message-passing substrate.

Both arms expose the same output contract: per-variable binary decision logits plus a verifier-confidence output. Capacity-reserve tensors may be used only to close exact parameter equality and are excluded from functional-state digests and optimizer updates, following the existing EXP-282 pattern.

The oracle advantage is therefore structural representation, not extra trainable parameter count. The oracle-information receipt is recorded explicitly.

### 2. Structure-dense neural world generator

Create a deterministic paired generator derived from named Stage-A RNG streams. Each batch contains:

- a surface/event representation available to both arms;
- variable states derived from the same event lineage;
- the oracle factor-incidence tensor available only to `oracle_cbrf`;
- binary per-variable targets;
- an exact batch digest binding every tensor and generation parameter.

Worlds contain multiple equality-connected components with independently anchored hidden values so solving requires identifying which observations/anchors constrain which variables. The surface lane contains the same facts without the compiled incidence matrix; the oracle lane receives the compiled structure.

Training uses the `augmentation` stream. Development evaluation uses the disjoint `evaluation` stream. Replicate IDs never overlap.

### 3. Accounted-compute contract

The two arms need exact trainable-parameter equality but need not execute identical operations. Each arm emits an analytical arithmetic ledger for the executed geometry. The development artifact records:

- exact total and functional parameter counts;
- exact world pairing and batch digests;
- per-arm accounted arithmetic FLOPs;
- hardware-profiler FLOPs claim = false;
- per-replicate verified utility divided by that arm's accounted FLOPs.

Both arms use the same declared maximum per-episode compute budget. If either arm exceeds the frozen budget, execution fails closed rather than clipping or silently changing geometry.

### 4. Paired development artifact

`run_exp277_paired_development(...)` emits `NLM-EXP-277-PAIRED-DEV-EVAL-V1` with:

- `evidence_level=EV-E2`;
- `decision=UNVERIFIED`;
- `confirmatory_ready=false`;
- canonical protocol/code lineage;
- matched initialization and parameter audit;
- oracle-information receipt;
- disjoint training/evaluation lineage;
- raw per-replicate metrics for both arms;
- primary relative utility gain summary;
- protected verified-solution-rate difference;
- immutable self-hash.

No decision rule is executed as a scientific promotion in development. The frozen 10% MESI is carried as metadata only.

### 5. Neural arm registry integration

Extend `build_neural_arm_registry(...)` with optional EXP-277 pair-audit/execution evidence, analogous to the existing EXP-282 path. Only after the matched arms and paired development artifact validate may the registry clear the two implementation blockers. Even then, EXP-277 remains development-only and the match court must not claim confirmatory readiness.

## Failure semantics

Fail before training/evaluation when any of these occur:

- protocol digest/arm drift;
- unequal trainable parameter count;
- oracle information leaks into ARCS inputs;
- training/evaluation replicate overlap;
- batch digest mismatch or reordered evaluation lineage;
- compute budget overflow;
- missing oracle-information receipt;
- artifact self-hash mismatch.

Scientific underperformance is not infrastructure failure and is retained in the artifact.

## Testing

TDD must cover:

- RED import failure before production modules exist;
- exact matched parameter budget;
- ARCS input cannot receive oracle incidence;
- deterministic batch regeneration and digest change under seed/geometry change;
- disjoint training/evaluation lineage;
- same paired batch lineage for both arms;
- compute ledger present and budget bounded;
- paired artifact remains EV-E2/UNVERIFIED/not confirmatory-ready;
- tamper + rehash cannot bypass semantic validation;
- neural arm registry clears only EXP-277 implementation blockers when valid evidence is supplied;
- existing EXP-282 ceremony and all current model-smoke tests remain green.

## Out of scope

- Confirmatory-open execution for EXP-277.
- Post-freeze challenge beacon materialization.
- EXP-278 learned constraint compiler.
- EXP-279 routing/hybrid execution.
- EV-E3/EV-E4/EV-E5 claims.
- Changing the 16M or 100M canonical model budgets.
