# Post-EXP-335 exploratory candidate: no-new-conflict half-space projection

Status: **POST-HOC EXPLORATORY DESIGN ONLY — NOT AN AUTHORIZED EXPERIMENT**

This note is derived from the non-decisional chunk-0 geometry audit of authoritative EXP-335 run `35445927525`. It MUST NOT modify, rerun, reinterpret, or replace EXP-335. It intentionally has no EXP number. A distinct preregistration may be created only after EXP-335 reaches its final immutable reducer.

## Empirical trigger

Authoritative chunk 0 showed two simultaneously true facts:

1. The registered `SUBSPACE_PROJECT_FULL32` projector almost completely removes negative dot mass on targets selected from the raw-negative set.
2. The remaining post-projection conflict mass is overwhelmingly caused by targets that were non-conflicting before projection and became conflicting afterward.

Observed chunk-0 values:

- 128 project updates.
- 1,763 raw-negative source/target relations.
- 1,692 selected projected targets.
- raw-negative mass: 138,388.1199169127.
- selected-target residual negative mass after projection: 0.025510004867792136.
- selected residual / raw-negative mass: ~1.84337e-7.
- 469 targets changed from raw dot >= 0 to post-projection dot < 0.
- new-conflict negative mass: 10,955.4641550265.
- new conflicts account for ~99.999767% of total post-projection negative mass.

The largest observed new conflicts are far beyond numerical noise. Therefore the phenomenon is structural, not merely tolerance error.

## Why the registered projector can leak conflict

Let the source gradient be `g` and let selected conflicting target gradients form the columns of `T`.

The registered rule is:

`g' = g - T pinv(T^T T) T^T g`

This is an orthogonal projection that drives dot products with the selected target subspace toward zero. It says nothing about target gradients outside the selected subspace. A previously positive target `u` can therefore satisfy:

`u^T g >= 0`

while after the projection:

`u^T g' < 0`.

This is exactly the leakage observed in chunk 0.

## Candidate geometry

For all 31 target gradients `t_i`, define the feasible half-spaces:

`t_i^T x >= 0` for every target `i`.

Choose the modified source gradient as the Euclidean projection of `g` onto the intersection of all 31 half-spaces:

`min_x 0.5 ||x - g||_2^2`

subject to:

`T^T x >= 0`.

This directly encodes the desired no-new-conflict invariant instead of only nulling the initially negative subset.

### Dual form

Using non-negative multipliers `lambda >= 0`, the optimum has:

`x = g + T lambda`.

Let:

- `G = T^T T`
- `b = T^T g`.

Then solve the small 31-variable convex QP:

`min_{lambda >= 0} 0.5 lambda^T G lambda + b^T lambda`.

After solving:

`x = g + T lambda`.

The large 10M-dimensional optimization is therefore reduced to a 31-variable dual problem plus the same gradient dot products already required by the court.

## Required invariants for a future preregistration

A future confirmatory intervention based on this design must freeze all of the following before execution:

- all 32 worlds and exact ordering;
- source update schedule and effort cycle;
- identical source and target RNG geometry;
- all 31 target gradients measured at the identical pre-update state;
- float64 Gram construction;
- deterministic dual solver and exact tie-breaking;
- solver tolerance, maximum iterations, and fail-closed non-convergence rule;
- gradient clipping after intervention, not before;
- no target filtering based on post-hoc outcomes;
- same A_FIXED 10,000,000-parameter reconstruction unless a separately authorized experiment changes the resident;
- all current scale/EXP-302/EXP-320 authorization flags remain false unless independently authorized.

## Solver requirements

The solver must be deterministic and auditable. It must not silently accept an infeasible or unconverged result.

At minimum, every update must emit:

- raw target-dot vector `b`;
- dual multipliers;
- active constraint set;
- primal post-dot vector `T^T x`;
- primal feasibility residual;
- dual feasibility residual;
- complementarity residual;
- source modification norm `||x-g||`;
- source preservation cosine between `x` and `g`;
- iteration count;
- solver termination code.

A non-finite value, solver non-convergence, or feasibility failure above the preregistered tolerance invalidates the arm.

## Required falsification courts

A future experiment must not assume the candidate is superior. It should attempt to falsify it against at least:

1. CONTROL — untouched source gradient.
2. SHAM — all 31 gradients measured, untouched source gradient applied.
3. REGISTERED_SUBSPACE — the exact EXP-335 rule.
4. NO_NEW_CONFLICT_QP — the half-space projection candidate.

The decisive geometry checks should include:

- number and mass of raw-negative target dots;
- number and mass of post-negative target dots;
- count of raw-nonnegative -> post-negative transitions;
- count of raw-negative -> post-nonnegative resolutions;
- update norm relative to the raw source update;
- per-world trajectory and final thresholds;
- exact CONTROL/SHAM state equality.

A no-new-conflict method fails its defining mechanism if any materially negative post-dot remains beyond the preregistered numerical tolerance.

## Counter-hypotheses that must remain live

Even if the QP eliminates gradient conflict geometrically, it may still fail scientifically:

- forcing all 31 instantaneous dot products non-negative may over-constrain learning;
- positive first-order dot products do not guarantee positive finite-step loss changes;
- target gradients can be noisy or locally misleading;
- minimizing gradient-space distance may not minimize parameter-space or function-space disturbance after AdamW state transformation and clipping;
- the constraint intersection can force a very small useful update;
- eliminating all conflict may reduce beneficial specialization;
- better instantaneous geometry may still worsen final 32-world retention.

Therefore success requires end-to-end evidence, not geometric cleanliness alone.

## Authority boundary

This document is post-hoc exploratory work. It grants no execution authority and no scale authority. It must remain separate from the immutable EXP-335 reducer. Only the final EXP-335 result may close the current experiment; a future experiment must receive its own preregistration, execution identity, frozen source, artifact chain, and reducer.
