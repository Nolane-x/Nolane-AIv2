# EXP-337 — C_NRS Frozen-Final Recurrent-Effort Sufficiency Court

**Scope:** design / preregistration only.  
**Implementation:** not authorized.  
**Execution:** not authorized.  
**Scale:** fixed at the existing 10,000,000-parameter substrate.

## Parent authority

EXP-336 is scientifically closed with:

- authoritative run `35481336946`, attempt `1`, SUCCESS
- final disposition `CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`
- evidence digest `c89f70f3555ad95892794821eddbcc158a2e7e9d7c4ef4fa02082a00acd6ccf7`
- closure digest `f33df20c0a1edea2751b4804cd5e51c7e34512841f2ad25aa1819ddad7a9666d`
- frozen successor scope `NEW_CAUSAL_HYPOTHESIS_PREREGISTRATION_ONLY`

EXP-337 therefore freezes one new causal hypothesis and nothing else.

## Causal question

EXP-336 ended with CONTROL `14/32` and PROJECT `18/32`, while the full aggregate Stage-A gate remained false. PROJECT improved the frozen foundation but did not establish Stage A.

EXP-337 asks:

> Can either exact frozen EXP-336 final checkpoint satisfy the complete Stage-A gate solely by increasing recurrent inference effort, with no learning or state mutation?

This separates an inference-compute sufficiency hypothesis from the need for a later learning/representation hypothesis.

## Exact frozen parents

Authoritative chunk7 artifact:

- artifact ID `10601286249`
- ZIP SHA-256 `5a12a2e59e399bec993ee1597ddbf4f85546e1d5de0b26635cb1d7def5a9666d`
- bundle digest `c5c72c4919077b0d1647939d4335cea79f579090af27fb2ff0a5e82e91d8cae8`

CONTROL:

- checkpoint SHA-256 `f616aea7f39c70e0a103ebe8f9b281a4487625d2b54418feea05e353c014290f`
- model-state digest `9d50a59a0cb6ed140e4c8bc9832ea1ca5efdd811226746a317e21f472e3b01e7`

PROJECT:

- checkpoint SHA-256 `a9e638f3029e4af05a685dd4cfba096da6107221ca3a31cb26e5422b4d394208`
- model-state digest `e122331caa2e6924c13d4c479574dc8d3e4e1de823adfc02604158ec3ad7fc38`

The independent 8/8 EXP-336 audit verified CONTROL=SHAM at every chunk. SHAM is therefore a parent-validity assertion, not a separate EXP-337 evaluation arm.

## Frozen intervention

Only recurrent inference effort changes:

`1, 2, 4, 8, 12, 16`

These are already accepted by the frozen parent forward primitive.

- effort `4`: mandatory exact EXP-336 reproduction boundary
- efforts `1,2`: diagnostic only
- efforts `8,12,16`: the only higher-effort rescue candidates

Every arm × effort cell begins from the exact frozen checkpoint. EXP-337 permits:

- 0 training updates
- 0 optimizer steps
- 0 weight updates
- 0 new parameters
- 0 allowed RNG advances
- no checkpoint mutation
- no persistent state carried between effort cells

## Evaluation contract

Teacher-forced evaluation uses the same 32 Stage-A worlds, tokenizer, answer encoding, loss, token accuracy and full-answer exact semantics as EXP-336.

Greedy evaluation preserves the existing `_diagnostic_generate` semantics:

- 96-token maximum
- argmax decoding
- stop on EOS
- unchanged invalid-token rule
- unchanged prompt/tokenizer
- no sampling

Future implementation may parameterize only the recurrent `effort` argument. Teacher-forced and greedy evaluation at one boundary must use the same effort.

## Effort-4 reproduction gate

Before any higher-effort result can be interpreted, effort 4 must exactly reproduce the authoritative EXP-336 final boundary for both CONTROL and PROJECT, including all 32 per-world vectors and aggregate metrics.

Any mismatch selects:

`EFFORT4_REPRODUCTION_MISMATCH`

and invalidates interpretation of efforts 8/12/16.

## Full Stage-A rescue gate

An arm is rescued only if, at one single effort in `8,12,16`, it passes the entire EXP-336 gate:

- all 32 worlds: token accuracy >= 0.99 and full-answer exact >= 0.90
- greedy exact >= 0.90
- aggregate token accuracy >= 0.99
- EOS correctness >= 0.95
- original-loss fraction <= 0.25
- invalid-output rate <= 0.01
- nonfinite events = 0

Pass count, family subset success, teacher-forced-only success, or a best-of-metrics composite cannot count as rescue.

## Frozen reducer

1. `INVALID_EXP337_FROZEN_EFFORT_COURT`
2. `PARENT_AUTHORITY_MISMATCH`
3. `EFFORT4_REPRODUCTION_MISMATCH`
4. `BOTH_FROZEN_FOUNDATIONS_RESCUED_BY_RECURRENT_EFFORT`
5. `CONTROL_FROZEN_FOUNDATION_RESCUED_BY_RECURRENT_EFFORT`
6. `PROJECT_FROZEN_FOUNDATION_RESCUED_BY_RECURRENT_EFFORT`
7. `FROZEN_FINAL_RECURRENT_EFFORT_NO_STAGE_A_RESCUE`

The transition lock permits only another preregistration/review scope after every outcome. It never opens implementation automatically.

## Interpretation boundary

A rescue means only that higher recurrent inference effort is sufficient for the exact frozen checkpoint under the complete Stage-A evaluation gate.

A no-rescue result means only that effort alone through the already-supported effort 16 is insufficient. It does not prove a unique replacement mechanism.

## Universal guards

The following remain false in this PR:

- EXP-337 implementation
- EXP-337 execution
- Stage-B design
- Stage-B implementation
- Stage-C implementation
- EXP-302 implementation
- EXP-320 implementation
- scaling
- 30M
- 100M

An explicit operator transition is required after a fully frozen, CI-green preregistration before any implementation work may begin.

## Frozen byte identities

Preregistration:

- canonical SHA-256 `2cfa555fae502f717658d8f401bf2fb33b9e68f97b0a69eb5395ce0c87f2a7a6`
- raw SHA-256 `0d5429fb0870e8176f3d383ae619f4a971345b27deeb339236d8e99aa00266a0`
- Git blob `04f0e84539753d0b0cb4ca54c3d9817a15f55319`

Post-result transition lock:

- canonical SHA-256 `f788044cf1384a54a179c2b545cf79715ab45ae621733e162bb9fe5cebdc56e2`
- raw SHA-256 `b9a35efc233f69a05a68a216101e72557ecd004068fb5833f0359aa27c97ea08`
- Git blob `be00da8a6cf5b4d283e4ba7e052306ae9fc9bf3e`

The machine-readable preregistration is authoritative over this explanatory note.
