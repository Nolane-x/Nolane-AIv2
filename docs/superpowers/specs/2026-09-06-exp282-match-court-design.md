# EXP-282 Match Court and Paired Partial-Observability Design

## Goal

Turn EXP-282 from an unmatched development component into a machine-audited paired neural comparison that closes the frozen parameter, observation-history, and accounted-compute match while remaining explicitly below confirmatory evidence.

## Authority

The implementation follows frozen `NLM-REASONING-STAGE-A-CONFIRMATORY-V1` EXP-282:

- `recurrent_hidden`: matched recurrent state without explicit belief representation;
- `explicit_belief`: explicit calibrated belief state over hidden world variables;
- primary endpoint: `grounded_decision_accuracy`;
- protected Brier guard: `explicit_belief <= recurrent_hidden + 0.02`;
- protected compute guard: relative accounted-FLOP difference `<= 0.05`;
- resource match: equal state/controller parameters, identical observation history, matched compute.

The frozen protocol bytes are not modified.

## Matched arm architecture

Both experiment-local arms use identical parameterized primitive topology: input projection, two-layer update transform, learned scale/bias, and binary decision head. The recurrent arm carries an opaque hidden vector across time. The explicit arm carries only binary belief logits across time; its hidden-shaped transform is transient and not cross-time state.

Both arms receive the exact same observation tensor. Functional parameters and total target parameters are equal. `capacity_reserve` remains non-functional and is excluded from optimization.

## Compute ledger

The ledger counts scalar linear multiplications, linear accumulations, bias additions, elementwise multiplications/additions, and exact nonlinear element/call signatures for SiLU, sigmoid, and tanh. Arithmetic FLOPs are reported separately from nonlinear primitive counts. The artifact explicitly states that these are analytical accounted FLOPs, not hardware-profiler FLOPs.

`accounted_flops_match` can be true only when both arithmetic counts and the complete primitive signature are identical for the declared `(timesteps, variables)` geometry.

## Partial-observability worlds

Latent binary targets come from the named `environment` RNG stream. Training observations use `augmentation`; held-out observations use `evaluation`. The same replicate therefore preserves the same latent target while changing visibility masks/noise across train/eval lanes.

For `timesteps >= 2`, each variable trajectory is forced to contain at least one visible and one hidden timestep. Batch artifacts retain observations, targets, visibility mask, seeds, geometry, and a deterministic digest.

## Paired neural execution

Both arms begin from byte-identical functional initialization derived from EXP-282 `model_init`. Separate AdamW optimizers use identical hyperparameters. Every training batch is shared between arms. Every held-out batch is shared between arms.

Raw held-out rows retain recurrent and explicit accuracy/Brier plus explicit-minus-recurrent contrasts. Aggregate development statistics never erase raw replicate lineage.

## Match Court states

Without matched-arm evidence, EXP-282 remains `BLOCKED`. With parameter/compute audit, it becomes `PARAMETER_AND_COMPUTE_MATCH_CLOSED` but remains blocked on execution lineage. With a valid paired partial-observability artifact, it becomes `PAIRED_PARTIAL_OBSERVABILITY_DEV_READY` but still remains `BLOCKED` for confirmatory execution.

The remaining blocker is the frozen confirmatory sample-size/paired-analysis freeze and post-freeze challenge execution. No development artifact may set `CONFIRMATORY_READY`, `PROMOTE_TO_NEXT_STAGE`, or claim EV-E3.

## Verification

Tests must cover parameter parity, primitive-operation parity, exact sequence-geometry scaling, deterministic stream separation, identical initialization, raw paired metrics, self-hash tamper detection, Match Court blocker transitions, CLI clean execution, and full repository regression.
