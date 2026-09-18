# EXP-324 — Family-Isolated Learnability Court

## Status

Preregistered diagnostic only. No EXP-324 training result exists yet.

EXP-324 follows sealed EXP-323R disposition `NO_REGISTERED_RESCUE`:

- repaired scientific run: `35354410137`;
- reducer recovery run: `35356470634`;
- final artifact: `10552276255`;
- final evidence digest: `a92d56a45151411a8fa4e0799949d72aed701f52ad9adae5fdb90983be09f051`.

EXP-323R demonstrated that from the exact selected step-2048 state, another 1024 joint updates at 5e-5 or at 2.5e-5 did not meet the teacher-forced floor and did not meet the registered progress threshold.

That rules out the two registered continuation rescues. It does **not** establish a 10M capacity ceiling.

## Question

The next unresolved question is:

> Can each task family become teacher-forced learnable when optimized in isolation with the same number of own-family updates it received during the failed joint continuation?

This distinguishes a family-local learnability problem from cross-family optimization competition/interference without changing model size.

## Immutable starting state

Every arm starts from the same materialized exact step-2048 reconstruction:

- artifact id `10547681681`;
- ZIP digest `c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621`;
- checkpoint SHA-256 `4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5`;
- model-state digest `4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc`;
- optimizer-state digest `9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79`;
- RNG-state digest `e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07`.

No replay selection is permitted.

## Four isolated arms

Exactly one arm exists for each frozen Stage-A family:

1. `algorithmic-sequence-transform`
2. `generator-heldout-abstract-transformation`
3. `iterative-grid-and-maze`
4. `language-sequence-control`

Each arm filters the original Stage-A root-0 order to its eight worlds, preserves that filtered order, and cycles it deterministically.

No world is selected by difficulty, loss, prior error, or post-hoc inspection.

## Matched update budget

The historical EXP-323R HOLD_5E5 control used 1024 joint optimizer updates over four equal eight-world families. Therefore each family received 256 training-world exposures.

EXP-324 gives each isolated family exactly **256 optimizer updates**, not 1024.

This prevents isolated arms from receiving four times the own-family exposure of the historical joint control.

Checkpoints are after local updates:

- 64  -> cumulative step 2112
- 128 -> cumulative step 2176
- 256 -> cumulative step 2304

The learning rate remains exactly `5e-5`. AdamW state, RNG state, weight decay, gradient clipping, tokenizer, objective and 10M architecture are unchanged at arm start.

## Primary floor

For the isolated family's eight worlds, at the same checkpoint:

- teacher-forced answer-token accuracy >= `0.99`;
- teacher-forced full-answer exact >= `0.90`.

Because each family has eight worlds, full-answer exact >= 0.90 means all eight must be exact.

## Historical matched-exposure control

The immutable joint HOLD_5E5 family metrics after 1024 joint updates / 256 implied exposures per family are bound in the machine preregistration. They are descriptive controls only and are not rerun or reselected.

## Dispositions

Exactly one is emitted:

1. `INVALID_FAMILY_ISOLATION`: malformed geometry, state/authority mismatch, non-finite event, missing or duplicated family/checkpoint.
2. `ALL_FAMILIES_ISOLATED_FIT`: all four families reach the primary floor in isolation.
3. `SOME_FAMILIES_ISOLATED_FIT`: at least one but not all families reach the floor.
4. `NO_FAMILIES_ISOLATED_FIT`: valid execution and no family reaches the floor.

`ALL_FAMILIES_ISOLATED_FIT` supports cross-family optimization competition/interference as a contributor, because each family can fit at matched own-family exposure when other-family gradients are removed. It does not uniquely identify the interference mechanism.

`SOME_FAMILIES_ISOLATED_FIT` localizes the remaining difficulty to one or more families while still allowing additional cross-family interference.

`NO_FAMILIES_ISOLATED_FIT` only says the registered isolation court did not reach the floor. It does not prove an architectural capacity ceiling.

## Secondary measurements

Each checkpoint also records greedy exact, answer-only loss, gradient/update norms and an all-32 evaluation to quantify transfer/forgetting. These cannot change the primary disposition.

## Authorization

EXP-324 cannot authorize implementation of EXP-302 or EXP-320, cannot authorize scale, and cannot authorize 30M or 100M. Every artifact must keep all five authorization flags false.
