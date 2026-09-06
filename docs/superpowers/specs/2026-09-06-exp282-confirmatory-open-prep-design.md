# EXP-282 Confirmatory-Open Preparation Design

## Goal

Freeze the analysis and sample-size inputs needed to begin EXP-282 confirmatory-open execution **without consuming confirmatory observations**, while preserving the frozen Stage-A V1 authority and keeping all preparation evidence at `EV-E2 / UNVERIFIED`.

## Authority

The authoritative experiment is `EXP-282` from `NLM-REASONING-STAGE-A-CONFIRMATORY-V1`. This preparation layer may read the protocol but must not modify it. It freezes the following existing commitments rather than inventing replacements after seeing confirmatory outcomes:

- primary endpoint: `grounded_decision_accuracy`, higher is better;
- MESI: absolute accuracy gain `0.03`;
- power target: `0.90`;
- familywise alpha: `0.05`;
- paired design;
- confirmatory sample-size range: `[32, 128]`;
- Brier guard: `explicit_belief <= recurrent_hidden + 0.02`;
- compute guard: relative accounted-FLOP difference `<= 0.05`;
- multiplicity family: `BELIEF_STATE`;
- analysis method: paired accuracy difference with bootstrap CI plus calibration guard.

## Inputs

The prep builder consumes:

1. the frozen `EXP-282` experiment specification;
2. one valid `NLM-EXP-282-PAIRED-DEV-EVAL-V1` development artifact;
3. one valid `NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1` artifact whose EXP-282 entry binds that exact paired-development artifact;
4. an analysis-code digest.

The pilot must contain at least 32 paired development replicates. Internal RNG streams are provenance and are not counted as additional observations.

## Pilot integrity

Every pilot replicate retains its raw recurrent/explicit accuracy and Brier values. The stored explicit-minus-recurrent contrasts must exactly agree with those raw values; derived contrast fields cannot be trusted independently. Replicate IDs must be unique. The held-out pilot lane must be `evaluation`, and resource matching must remain closed.

The prep artifact binds the complete paired-development artifact digest, Match Court registry digest, protocol digest, analysis-code digest, and a digest of the paired effect vector plus replicate IDs.

## Sample-size freeze

For the first executable confirmatory-open preparation, sample-size planning uses a transparent paired normal approximation from the development pilot SD:

`ceil(((z_alpha + z_power) * paired_sd / MESI)^2)`

where `z_alpha` uses a one-sided lower-bound alpha of `0.05` and `z_power` uses power `0.90`. The result is then raised to the frozen minimum of 32 when necessary.

This is an explicit V1 engineering planning assumption, not a neural result. The artifact stores `z_alpha`, `z_power`, formula text, pilot SD, and the unclamped required n so reviewers can reproduce the calculation.

**No maximum truncation is permitted.** If the unclamped requirement exceeds 128, the status is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, `confirmatory_n` remains null, and no confirmatory replicate IDs are reserved. A high-variance pilot therefore blocks execution rather than being converted into an underpowered confirmatory run.

## Data separation

When the sample-size requirement is feasible, confirmatory replicate IDs are reserved strictly after the highest pilot replicate ID. Pilot IDs and confirmatory IDs must be disjoint. The prep artifact states `pilot_reuse_forbidden=true` and `confirmatory_data_consumed=false`.

The prep phase does not materialize confirmatory observations, does not run either arm on reserved confirmatory worlds, and does not derive post-freeze challenge seeds. Challenge randomness remains a later future-beacon operation.

## Result states

Preparation has two statuses:

- `CONFIRMATORY_OPEN_PREPARED`: the frozen development pilot is large enough and the sample-size requirement fits `[32,128]`;
- `NOT_READY_VARIANCE_EXCEEDS_MAX_N`: the pilot SD implies an n above the frozen maximum.

Both remain `EV-E2 / UNVERIFIED`. Neither status promotes the neural claim.

## Anti-laundering rules

The validator rejects:

- EV-E3+ evidence labels or scientific promotion decisions;
- confirmatory-data-consumed flags;
- MESI, power, alpha, endpoint, Brier guard, compute guard, analysis-method, or multiplicity drift;
- missing provenance digests;
- pilot n below 32;
- pilot/confirmatory replicate overlap;
- a prepared n that differs from the frozen sample-size rule;
- a not-ready artifact that nevertheless reserves confirmatory IDs;
- self-hash tampering.

## Out of scope

This phase does **not** run confirmatory-open observations, does not evaluate the frozen decision rule, does not derive a future challenge beacon, and does not claim EV-E3. Those require later separately frozen execution and evidence packets.
