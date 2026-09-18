# EXP-323 — A_FIXED Second-Decay Convergence Intervention

## Status

Design and preregistration only. No EXP-323 training result exists until the exact source is sealed and an authoritative one-shot workflow is dispatched.

EXP-323 follows authoritative EXP-322:

- run: `35339004168`
- disposition: `PARTIAL_CONTINUATION_PROGRESS`
- final evidence digest: `1e3ea213718d1612a6370c0fb4124c869733747dc9992ca58345d16dcfb5aee2`
- DECAY_5E5 arm evidence digest: `78c4c44ae168a4a186ceb8040e239501804263e9b45737afdee25fb465c038ce`
- DECAY step-2048 model-state digest: `4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc`
- optimizer-state digest: `9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79`
- RNG-state digest: `e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07`

EXP-323 does not authorize EXP-320, 30M, 100M, or any scale change.

## Why this court exists

EXP-322 showed two facts simultaneously:

1. continuing at the inherited `1e-4` learning rate improved the selected A_FIXED checkpoint but did not reach the teacher-forced floor;
2. the `5e-5` arm produced substantially stronger step-2048 metrics — token accuracy `0.9577464788732394`, full-answer exact `0.6875`, greedy exact `0.6875`, answer-only loss `0.4435557168891316` — but also remained below the floor.

The next unresolved question is therefore narrower than "is 10M too small?":

> From the exact EXP-322 DECAY state at step 2048, is another 1024-step budget at 5e-5 sufficient, or is a second halving to 2.5e-5 required?

Neither outcome may be interpreted as evidence for or against larger model capacity.

## Exact reconstruction before intervention

EXP-322 did not persist a trainable checkpoint artifact at step 2048. EXP-323 therefore must reconstruct that state deterministically from the immutable EXP-319 step-1024 checkpoint.

Before any optimizer step greater than 2048, the runtime must:

1. download and verify EXP-322 final evidence artifact identity and ZIP digest;
2. download and verify the immutable EXP-319 selected checkpoint artifact;
3. load exact model, AdamW state and CPU RNG state from step 1024;
4. replay exactly the EXP-322 `DECAY_5E5` trajectory for global steps 1024 through 2047;
5. require the reconstructed step-2048 digests to equal all three authoritative EXP-322 anchors:
   - model `4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc`;
   - optimizer `9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79`;
   - RNG `e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07`.

Any mismatch yields `INVALID_INTERVENTION`. No post-2048 update may occur after a replay mismatch.

## Immutable population and model

Exactly the same 32 Stage-A root-0 worlds are reused: four families, eight examples each. Population role remains tuning/sanity only.

The resident model remains exactly 10,000,000 trainable parameters, CPU-only, A_FIXED. Architecture, tokenizer, data, canonical answers, answer-only objective, world ordering, AdamW optimizer family, optimizer moments, weight decay, gradient clip and effort cycle are immutable.

## Arms

Both arms clone the exact reconstructed step-2048 state.

### HOLD_5E5

Continue at learning rate `5e-5` for at most 1024 further optimizer steps. This isolates additional budget at the already-reduced learning rate.

### DECAY_2P5E5

Before the first step greater than 2048, change only every AdamW parameter-group learning rate from `5e-5` to exactly `2.5e-5`. Preserve all optimizer moments and every other state field.

This is one preregistered second-decay intervention, not a learning-rate sweep.

## Checkpoints

Evaluate both arms at cumulative global steps:

- 2304
- 2560
- 3072

No preferred arm may stop early after seeing a favorable result. All checkpoints must be emitted unless a fail-closed invalidity occurs.

## Primary floor

At gating effort 4 over all 32 worlds, both conditions must hold at the same checkpoint:

- teacher-forced answer-token accuracy >= `0.99`;
- teacher-forced full-answer exact >= `0.90`.

Greedy exact is secondary and cannot independently trigger sufficiency.

## Baseline and material-progress rule

The immutable EXP-322 DECAY step-2048 baseline is:

- token accuracy `0.9577464788732394`;
- full-answer exact `0.6875`;
- greedy exact `0.6875`;
- answer-only loss `0.4435557168891316`.

Because only `0.0322535...` token-accuracy headroom remains before the primary token floor, EXP-323 defines material non-floor progress as either:

- token accuracy gain >= `0.02`; or
- full-answer exact gain >= `0.125` (four additional full answers out of 32).

These thresholds are descriptive only and cannot authorize scale.

## Decision order

Exactly one disposition is emitted.

1. `INVALID_INTERVENTION`: any parent/replay/invariant mismatch, malformed geometry, missing/duplicate checkpoint, non-finite primary metric, or non-finite training event.
2. `REDUCED_LR_BUDGET_SUFFICIENT`: HOLD_5E5 reaches both primary floors at any registered checkpoint. This has precedence even if the second-decay arm also passes.
3. `SECOND_DECAY_SUFFICIENT`: HOLD_5E5 never reaches both floors while DECAY_2P5E5 does.
4. `CONTINUATION_PROGRESS`: neither arm reaches both floors but at least one clears a registered material-progress threshold relative to the immutable step-2048 baseline.
5. `NO_REGISTERED_RESCUE`: valid execution, neither floor pass and no registered material progress.

## Explicit non-claims

Even `NO_REGISTERED_RESCUE` cannot establish that 10M lacks capacity, that a larger model would work, or that architecture/scaling is the next correct intervention. A capacity court requires a separate preregistration.

All of the following remain hard-coded false in every artifact:

- `exp302_implementation_authorized`
- `exp320_implementation_authorized`
- `scale_authorized`
- `authorized_30m`
- `authorized_100m`

## Implementation order

1. freeze this design and machine preregistration;
2. freeze the pure reducer and adversarial decision tests;
3. implement replay authority and require exact step-2048 state reproduction;
4. implement the two post-2048 continuation arms;
5. implement arm/final evidence and execution identity binding;
6. implement sealed workflow and complete dedicated court;
7. require same-head generic CI + inherited EXP-301 court + scope audit;
8. create marker-only execution identity;
9. register a byte-identical workflow on main;
10. dispatch exactly one authoritative EXP-323 run.

No scientific result may be inspected before steps 1-9 are complete.
