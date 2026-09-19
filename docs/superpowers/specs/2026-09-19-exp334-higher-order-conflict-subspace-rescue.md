# EXP-334 — Higher-Order Conflict-Subspace Rescue Court

EXP-333 established `COMPLETE_8WORLD_PAIR_LATTICE_RESCUE_NO_REGRESSION` across all 28 unordered pairs of iterative-grid-and-maze worlds 0..7. It found two CONTROL failures, `P02` and the newly exposed `P06`, and the exact pairwise conflicting-gradient projection rescued both with no pair regression.

EXP-334 asks the next bounded question: can the same causal idea extend beyond pairs to the higher-order groups that historically exposed interference?

## Frozen groups

The court uses seven previously registered higher-order groups:

- triples: `T012`, `T013`, `T023`, `T123`
- quartets: `Q0123`, `Q4567`
- octet: `O01234567`

Historical CONTROL behavior is frozen before execution:

- failures: `T012`, `T023`, `Q0123`, `O01234567`
- passing controls: `T013`, `T123`, `Q4567`

The triple and `Q0123` anchors come from immutable EXP-327 replay evidence. `Q4567` and `O01234567` come from the immutable EXP-326 iterative-family artifact. Historical cross-run model/optimizer bitwise equality is not required; members, final token metrics, final exact metrics, RNG and pass/fail must reproduce. Within EXP-334, every SHAM must equal CONTROL exactly in metrics/model/optimizer/RNG.

## Multi-target projection

For one source update, all other members of the same group are evaluated at the exact same pre-update state and effort.

For each target `t`, compute `g_s · g_t`. Targets with negative dot and target norm squared above `1e-24` form the conflicting target set. Let the conflicting target gradients be columns of `T`. EXP-334 applies:

`g_projected = g_s - T pinv(T^T T) T^T g_s`

with the small Gram system evaluated in float64 and preregistered pseudoinverse tolerance.

When exactly one target is selected, this reduces algebraically to the EXP-330 pairwise projection rule.

The SHAM arm performs the same measurements but applies the untouched source gradient and restores the source RNG state before the optimizer step.

## Frozen geometry

- A_FIXED exactly 10,000,000 trainable parameters
- immutable step-2048 reconstruction
- AdamW
- LR `5e-5`
- weight decay `0.01`
- clip `1.0`
- 32 exposures/world
- world-local effort cycle `1,2,4,8`
- ascending world index within each exposure
- CONTROL / SHAM / SUBSPACE_PROJECT

## Strongest positive

`HIGHER_ORDER_SUBSPACE_RESCUE_NO_REGRESSION` requires:

1. every historical CONTROL anchor reproduces;
2. all seven SHAM arms exactly match CONTROL;
3. every CONTROL-failing higher-order group triggers subspace projection;
4. every CONTROL-failing group passes under SUBSPACE_PROJECT;
5. every CONTROL-passing group remains passing;
6. no nonfinite event occurs.

This is a bounded higher-order rescue claim only. It does not establish all 56 triples or all 70 quartets of the eight worlds, cross-family generalization, persistent state, online plasticity, EXP-302/EXP-320 authorization, or scale authorization.

Preregistration canonical digest: `cd973c9724b27ad4d881de2211522cb315c9a34275f613395c69b675d8c3fdb3`  
Preregistration exact JSON SHA-256: `5d735e545344ec305649a12559e793498a0c72ee326cdba507805fc8543a8d39`
