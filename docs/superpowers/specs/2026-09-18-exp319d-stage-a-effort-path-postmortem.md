# EXP-319D1 — Stage-A Effort-Path Postmortem

Status: post-hoc read-only diagnostic / no scientific authorization

## 1. Authority and purpose

EXP-319 has an authoritative recovered disposition:

`TRAINING_STACK_NOT_LEARNABLE`

The disposition is bound to authoritative scientific run `35311529822`, frozen source
`2002a42322c7b919c3c9dc3da7d9cb0f546d4431`, repaired execution marker
`124584061616ab0864355219645ed00c1298c575`, Stage-A selection authority digest
`39e5e846dbe87f3d5b34e438f0c8864626caa7049148fac934f2c893032e623f`, and recovered
final evidence digest `4be2850f6c1d633af4cb6687e379c0011aaf8281075bea79821d0f2e038ffb07`.

EXP-319D1 does not reopen, replace, weaken, or reinterpret that result. It performs no
training and no parameter updates. Its only purpose is to explain a concrete ambiguity in
the failed Stage-A sanity gate before any new learnability experiment is designed.

## 2. Observed evidence motivating the diagnostic

The sealed Stage-A selection artifact reports:

- A_FIXED: LR 1e-4, loss 343.3998 -> 2.3283, token accuracy 0.7711,
  exact match 0.28125, EOS 1.0, invalid output 0.0, non-finite events 0;
- B_LOOP_SIMPLE: LR 1e-4, loss 292.6865 -> 1.4326, token accuracy 0.7324,
  exact match 0.3125, EOS 0.71875, invalid output 0.0, non-finite events 0;
- C_NRS_CORE: LR 3e-4, loss 164.5518 -> 2.1496, token accuracy 0.7007,
  exact match 0.25, EOS 1.0, invalid output 0.0, non-finite events 0.

The frozen Stage-A floor requires token accuracy >= 0.99 and exact match >= 0.90.
Therefore the control A_FIXED fails and the frozen reducer correctly emits
`TRAINING_STACK_NOT_LEARNABLE`.

However, loss reduction is large and numerics are stable. That evidence does not by itself
distinguish at least three explanations:

1. the 1024-step / LR training budget is insufficient for memorization;
2. teacher-forced learning exists but greedy autoregressive decoding suffers an error
   cascade;
3. training optimizes a mixture of compute paths while the gate observes only one path.

The third explanation is structurally testable without retraining. Frozen training cycles
effort `1,2,4,8`, while Stage-A evaluation is fixed at effort `4`. In A_FIXED, effort is
the number of stateless restarts; in recurrent arms it is the number of loops. These are
different forward computations, not metadata-only labels.

## 3. Diagnostic question

For each already-selected Stage-A step-1024 checkpoint, what are the teacher-forced and
greedy generation metrics when the *same frozen weights* are evaluated independently at
efforts `1,2,4,8`?

The diagnostic reports measurements only. It does not introduce a post-hoc pass/fail
threshold and does not choose a replacement scientific outcome.

## 4. Inputs

Only immutable evidence from run `35311529822` is allowed:

- `exp319-stage-a-selection`, artifact id `10533822077`, GitHub artifact digest
  `sha256:ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559`;
- selected step-1024 checkpoint artifact for A_FIXED / LR 1e-4;
- selected step-1024 checkpoint artifact for B_LOOP_SIMPLE / LR 1e-4;
- selected step-1024 checkpoint artifact for C_NRS_CORE / LR 3e-4.

The analyzer must verify the selection authority, receipt, arm, root, LR, cumulative step,
source commit, training-contract digest, checkpoint model-state digest, and parent lineage
before evaluating.

## 5. Immutable execution rules

The analyzer:

- runs on CPU only;
- loads the frozen checkpoint in evaluation mode;
- performs zero optimizer construction and zero backward passes;
- evaluates exactly the 32 frozen Stage-A worlds;
- evaluates efforts exactly `1,2,4,8`;
- uses the frozen byte tokenizer, answer-only loss, deterministic verifier, and greedy
  decoder semantics;
- includes EOS in teacher-forced answer-token accuracy exactly as EXP-319 does;
- uses at most 96 generated tokens exactly as EXP-319 does;
- records family-level exact match as well as aggregate metrics;
- recomputes the model-state digest after all evaluations and requires it to equal the
  pre-evaluation digest.

Any parameter-state change makes the diagnostic invalid.

## 6. Output schema

One result is emitted per arm. It contains:

- fixed source/run/selection/checkpoint identities;
- pre/post model-state digest;
- one metric record per effort:
  - answer-only loss;
  - answer-token accuracy;
  - exact match;
  - family-balanced exact match;
  - per-family exact match;
  - EOS correctness;
  - invalid-output rate;
  - number of nonzero exact families;
- the effort with the highest exact match under deterministic tie-breaking;
- the effort-4 record copied as the frozen gate reference;
- the exact-match and answer-token-accuracy deltas from effort 4 for every effort;
- a canonical SHA-256 evidence digest.

No field is named `pass`, `authorize`, `winner`, or `replacement_disposition`.

## 7. Interpretation boundary

A different effort outperforming effort 4 would be evidence that the frozen mixed-effort
training path and fixed-effort gate are not behaviorally interchangeable. It would *not*
prove that EXP-319 should have passed and would not permit selecting the better effort
post hoc.

If all efforts are similarly weak, the evidence shifts attention toward budget/LR,
teacher-forcing/autoregressive exposure, data geometry, or broader training-stack issues.

Either result may justify a new preregistered 10M learnability experiment. Neither result
authorizes EXP-320 implementation or scale-up.

## 8. Scientific locks

EXP-319D1 cannot authorize or imply authorization for:

- EXP-302 implementation;
- EXP-320 implementation;
- 30M;
- 100M;
- any architecture ranking;
- any reinterpretation of EXP-301 `KILL_H_RD_01`;
- any reinterpretation of EXP-319 `TRAINING_STACK_NOT_LEARNABLE`.

All such flags remain false.

## 9. Completion gate

EXP-319D1 is complete only when:

1. unit tests verify strict evidence binding, deterministic metrics/digest behavior, and
   mutation detection;
2. repository CI is green on one exact head;
3. the read-only workflow evaluates all three selected checkpoints from run
   `35311529822`;
4. all three output artifacts are independently inspected;
5. a combined evidence note records the measured results and limits;
6. no scientific source, EXP-319 preregistration, frozen reducer, or sealed history was
   modified.

This diagnostic is disposable evidence plumbing. It must not be merged into a scientific
lineage merely because it produces an interesting post-hoc result.
