# EXP-326 — Balanced Multi-World Interference Breakpoint Court

## Status

Design and preregistration only. No EXP-326 scientific result exists until the exact source is frozen, marker-only sealed, registered byte-identically on main, and dispatched once.

EXP-326 follows the sealed EXP-325 result:

- authoritative run: `35406123945`
- disposition: `ALL_WORLDS_SINGLE_FIT`
- passed worlds: `32/32`
- final evidence digest: `0ce6a8bd4f96864bf754b22fc7c9b95552f6081225a4c305394d92ed39e7cbc2`
- final artifact id: `10572536792`

Every frozen Stage-A world can be memorization-fit independently from the exact same immutable step-2048 reconstruction. In the EXP-325 artifacts all 32 worlds already satisfy token accuracy = 1.0 and full-answer exact = 1.0 by local update 8.

## Why EXP-324 is not a clean k=8 endpoint

The training helper selects recurrent effort using `effort_for_training_step(global_step)` with the cycle `(1,2,4,8)`.

EXP-324 trained eight family worlds in round-robin order using global step. Because eight is a multiple of the four-effort cycle, an individual world can repeatedly occupy the same effort residue. EXP-325 single-world isolation instead traverses the full effort cycle for every world.

Therefore EXP-324 remains valid as the court it preregistered, but it does **not** cleanly isolate group-size interference from per-world effort assignment. EXP-326 corrects this confound.

## Causal question

> When per-world optimizer exposure and per-world effort exposure are held equal to the successful EXP-325 single-world court, at what registered group size does simultaneous memorization first fail?

The registered sizes are 2, 4, and 8 worlds, nested within each family.

## Frozen groups

For every one of the four Stage-A families, create independent clones from the same immutable step-2048 reconstruction.

Pairs:

- P01 = [0,1]
- P23 = [2,3]
- P45 = [4,5]
- P67 = [6,7]

Quartets:

- Q0123 = [0,1,2,3]
- Q4567 = [4,5,6,7]

Octet:

- O01234567 = [0,1,2,3,4,5,6,7]

This yields 28 independent group fits total: 16 pairs, 8 quartets, 4 octets.

## Exposure-matched and effort-balanced training

Every world receives exactly 32 optimizer exposures, exactly matching EXP-325.

For world-local exposure index `j = 0..31`, every world receives effort:

`(1,2,4,8)[j mod 4]`

Thus every world receives exactly eight updates at effort 1, eight at effort 2, eight at effort 4, and eight at effort 8.

Within an exposure round, worlds are updated in ascending registered index.

Total optimizer updates are therefore:

- pair: 64
- quartet: 128
- octet: 256

Evaluation occurs after every world has received 8, 16, and 32 exposures.

No group shares optimizer state with any other group.

## Primary pass rule

A registered group passes if at any registered checkpoint **every member world** simultaneously satisfies:

- teacher-forced answer-token accuracy >= 0.99;
- teacher-forced full-answer exact >= 0.90.

Aggregate group averages cannot hide an individual failing world.

## Nested monotonicity

The grouping is deliberately nested.

A non-monotonic violation occurs if:

- Q0123 passes while P01 or P23 fails;
- Q4567 passes while P45 or P67 fails;
- O01234567 passes while Q0123 or Q4567 fails.

Such a result receives its own disposition rather than being forced into a scalar breakpoint.

## Decision order

1. `INVALID_BREAKPOINT_COURT`
2. `NONMONOTONIC_GROUP_FIT`
3. `PAIR_LEVEL_BREAK_PRESENT`
4. `QUARTET_LEVEL_BREAK_PRESENT`
5. `OCTET_LEVEL_BREAK_PRESENT`
6. `NO_BREAK_THROUGH_EIGHT`

The decision identifies the earliest registered monotonic failure scale, not a universal architecture limit.

## Scientific boundary

No architecture, tokenizer, objective, parameter count, reconstruction state, optimizer family, inherited moments, LR, weight decay, gradient clipping, or world identities change.

All scale / EXP-302 / EXP-320 authorization remains false.

If interference is observed, later experiments may localize gradient conflict, retention/catastrophic forgetting, or representation collision. EXP-326 itself does not choose among those mechanisms.
