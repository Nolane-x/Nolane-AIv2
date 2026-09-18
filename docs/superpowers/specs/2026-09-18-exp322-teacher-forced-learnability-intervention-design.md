# EXP-322 — A_FIXED Teacher-Forced Learnability Intervention

## Status

Design and preregistration only. No EXP-322 training result exists until the exact source is sealed and the authoritative workflow is dispatched.

EXP-322 follows the sealed EXP-321 result:

- authoritative run: `35331243762`
- EXP-321 disposition: `TEACHER_FORCED_FOUNDATION_INSUFFICIENT`
- EXP-321 evidence digest: `5f3bd6cbacef0af4883e33f14bafd56b49a307df88641e837bb9ecdf004d5091`
- selected EXP-319 checkpoint: `A_FIXED@1e-4`, root 0, cumulative step 1024
- selected model-state digest: `18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb`

EXP-322 does not rescue or reinterpret EXP-301/319/321, does not authorize EXP-320, and does not authorize 30M/100M scaling.

## Scientific question

The selected 10M supervised control optimized substantially but remained far below the teacher-forced floor at step 1024. EXP-321 ruled out effort mismatch and unused-vocabulary competition as dominant explanations.

EXP-322 asks one narrower question:

> Is the missing teacher-forced learnability primarily explained by insufficient continuation budget at the inherited learning rate, or by the inherited learning rate remaining too large after step 1024?

No architecture, tokenizer, data, objective, task population, optimizer family, optimizer moments, initial checkpoint, RNG state, weight decay, gradient clip, or effort cycle may change.

## Immutable parents

The experiment is bound to:

- EXP-321 run `35331243762`
- EXP-321 evidence digest `5f3bd6cbacef0af4883e33f14bafd56b49a307df88641e837bb9ecdf004d5091`
- EXP-321 artifact id `10541366011`
- EXP-319 selected checkpoint artifact id `10534351390`
- EXP-319 checkpoint ZIP SHA-256 `ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3`
- EXP-319 receipt artifact digest `d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280`
- EXP-319 model-state digest `18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb`

Checkpoint substitution is forbidden.

## Population

Exactly the 32 frozen Stage-A root-0 worlds are reused:

- 4 task families
- 8 examples per family
- identical prompts
- identical canonical answers
- identical deterministic world order

This remains tuning/sanity evidence and is not generalization evidence.

## Starting state

Both arms are cloned from the exact same step-1024 checkpoint bundle, including:

- model parameters
- AdamW optimizer state and moments
- CPU RNG state
- data-order authority
- model initialization identity

Both arms begin at global step 1024. The starting state must reproduce the sealed parent identities before any optimizer update occurs.

## Intervention arms

### HOLD_1E4

Continue for at most 1024 additional optimizer steps with learning rate `1e-4`.

This is the pure budget intervention. No optimizer field is changed after loading the parent bundle.

### DECAY_5E5

Clone the same parent bundle and, before the first continuation update, set every AdamW parameter-group learning rate to exactly `5e-5`.

All optimizer moments remain inherited. Nothing else changes.

This is a single preregistered learning-rate intervention, not an LR search.

## Training geometry

For both arms:

- resident trainable parameters: exactly 10,000,000
- CPU only
- AdamW
- weight decay `0.01`
- gradient clip norm `1.0`
- answer-only loss unchanged
- byte tokenizer unchanged
- effort cycle `(1,2,4,8)` unchanged and indexed by the inherited global step
- world ordering unchanged
- maximum cumulative step: `2048`
- evaluation checkpoints: `1280, 1536, 2048`

No arm may stop early because a preferred result appears. All preregistered checkpoints must be produced unless the arm becomes invalid through non-finite state or authority failure.

## Primary measurements

At each checkpoint and at gating effort 4, over all 32 worlds:

1. teacher-forced answer-token accuracy, including EOS;
2. teacher-forced full-answer exact rate.

The registered teacher-forced floor is unchanged:

- token accuracy >= `0.99`
- full-answer exact >= `0.90`

Both must hold at the same checkpoint.

## Secondary measurements

Record, but do not use to declare a teacher-floor pass:

- greedy exact
- EOS correctness
- invalid-output rate
- answer-only loss
- gradient norm before clipping
- parameter-update norm ratio
- non-finite event count
- per-family teacher-forced token accuracy
- per-family teacher-forced full-answer exact

Greedy improvement alone cannot change the primary disposition.

## Baseline

The sealed step-1024 EXP-321 baseline is:

- teacher-forced token accuracy: `0.7711267605633803`
- teacher-forced full-answer exact: `0.28125`
- greedy exact: `0.28125`

## Decision rules

Exactly one disposition is emitted.

### INVALID_INTERVENTION

Any parent identity mismatch, wrong world geometry, missing or duplicate arm/checkpoint, changed invariant, non-finite primary metric, or non-finite training event invalidates the intervention.

### BUDGET_INSUFFICIENCY_EVIDENT

`HOLD_1E4` reaches both teacher-forced floors at any registered checkpoint.

This has precedence even if `DECAY_5E5` also passes, because the unchanged inherited learning rate was sufficient once additional budget was supplied.

### LR_SCHEDULE_INSUFFICIENCY_EVIDENT

`HOLD_1E4` never reaches both floors, while `DECAY_5E5` reaches both at a registered checkpoint.

### PARTIAL_CONTINUATION_PROGRESS

Neither arm reaches both floors, but at least one arm improves over the sealed step-1024 baseline by either:

- teacher-forced token accuracy >= `+0.10`, or
- teacher-forced full-answer exact >= `+0.25`.

This is descriptive progress only and does not authorize a new model or scale.

### NO_REGISTERED_RESCUE

The intervention is valid, neither arm reaches the teacher-forced floor, and neither arm clears a registered partial-progress threshold.

This means only that the two registered continuation interventions failed to demonstrate rescue.

## Explicit non-claims

Even `NO_REGISTERED_RESCUE` cannot establish:

- architectural capacity exhaustion;
- that 10M is too small;
- that a larger model would work;
- NRS superiority or failure;
- held-out generalization;
- permission to modify EXP-320;
- permission to scale.

A capacity or architecture experiment would require a separate preregistration.

## Authorization boundary

Every EXP-322 artifact must hard-code all of the following to false:

- `exp302_implementation_authorized`
- `exp320_implementation_authorized`
- `scale_authorized`
- `authorized_30m`
- `authorized_100m`

## Implementation order

1. freeze this design and machine-readable preregistration;
2. implement the pure reducer and adversarial contract tests;
3. implement continuation-state authority checks;
4. implement teacher-forced evaluator;
5. implement two-arm continuation runtime without modifying EXP-319 frozen source;
6. implement workflow and evidence reducer;
7. run full repository regression;
8. freeze exact source/tree/workflow/preregistration identity;
9. create marker-only execution identity;
10. dispatch exactly one authoritative EXP-322 execution.

No scientific outcome may be inspected before steps 1–9 are complete.
