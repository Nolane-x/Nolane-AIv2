# EXP-289 Confirmatory Gate-A Design

## Purpose

Advance frozen Stage-A experiment EXP-289 / H-CONFLICT-01 from DEVELOPMENT EV-E2 machinery toward a scientifically valid confirmatory path without changing `protocols/stage_a_v1.json` or its digest.

## Frozen authority

EXP-289 compares `no_nogood` against `local_nogood`. The primary endpoint is `repeat_dead_end_rate` (lower is better), with frozen MESI = 25% relative reduction. Protected endpoints are `valid_state_overprune_rate <= 0.005` and `local_nogood verified_solution_rate >= no_nogood - 0.01`. The global paired sample-size plan remains min n=32, max n=128, power=0.90, familywise alpha=0.05. The multiplicity family is `LOCAL_NOGOOD`. Challenge randomness may only be derived after protocol/code/config/evaluator/analysis freeze from a future public beacon.

## Gate-A estimand and denominator policy

For each paired DEVELOPMENT replicate, derive arm-level repeat-dead-end rates only from predeclared evaluator opportunity counts. Episodes with zero repeat opportunities remain in raw evidence and safety accounting but contribute no RDER ratio denominator. A replicate is planning-eligible only when the `no_nogood` baseline denominator is finite and strictly positive and both arm rates are finite. No epsilon rescue is permitted.

The planning effect is the paired relative reduction `(RDER_no_nogood - RDER_local_nogood) / RDER_no_nogood`. Gate A estimates paired effect variability from DEVELOPMENT observations only. It freezes confirmatory n before any challenge materialization. If the implied n exceeds 128, the result is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`; it must not be forced READY by changing MESI, denominator handling, alpha, or thresholds after seeing pilot results.

## Safety boundary

Gate A does not execute the scientific decision rule. It verifies that DEVELOPMENT evidence preserves episode/problem-local scope, no cross-episode/problem reuse, exact subset matching, evaluator-truth separation from causal arm actions, charged memory storage/retrieval work, and the two frozen protected endpoints. Development observations cannot be reclassified as confirmatory evidence.

`valid_state_overprune_rate` must be reconstructed from raw prune receipts/evaluator audit rather than trusted as a top-level summary. Zero-opportunity episodes remain eligible for over-prune and solution safety auditing even when excluded from the primary RDER denominator.

## Authority chain

The confirmatory program is staged:

1. RDER-specific Gate-A preparation and fail-closed validator.
2. Authoritative DEVELOPMENT geometry declaration and exact execution evidence.
3. Exact trained-state checkpoint/reconstruction court for both arms.
4. Execution authorization that binds protocol, code tree, geometry, DEVELOPMENT artifact, checkpoint identity, frozen analysis, and challenge contract.
5. Immutable pre-beacon Gate-A seal.
6. Future public beacon receipt and domain-separated challenge seed.
7. Hidden confirmatory world materialization with no operator raw-seed override.
8. Raw-before-analysis Gate-B executor and frozen bootstrap/safety analysis.
9. EV-E3 decision only for a valid real ceremony; TEST-ONLY CI ceremonies remain EV-E2 / UNVERIFIED.

## Scientific decisions

A real valid Gate B may promote only if the one-sided lower confidence bound on relative RDER reduction is at least 0.25 and all protected guards pass. It must kill the frozen subsystem comparison if the reduction fails the frozen MESI or over-prune exceeds 0.5%; solution-floor failure also prevents promotion. Integrity failures remain `INVALID_RUN`, never relabeled as scientific kills.

## Non-claims

Any result applies only to the frozen episode-local exact-nogood experiment family. It does not establish cross-problem learned-clause transfer, lifelong lemma economy, unrestricted symbolic memory benefit, or the integrated NLM Stage-B/Stage-C claim.
