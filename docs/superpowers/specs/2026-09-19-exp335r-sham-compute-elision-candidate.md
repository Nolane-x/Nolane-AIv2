# EXP-335R contingency: SHAM compute-elision candidate

Status: **NON-AUTHORITATIVE CONTINGENCY ONLY**.

This branch is prepared while authoritative EXP-335 run `35445927525` remains in progress. It must not replace, pool with, or reinterpret that run. Use of this repair requires a separate explicit execution identity and a new sealed run only if the authoritative court becomes operationally unusable (for example, timeout or infrastructure failure).

## Problem

The sealed EXP-335 implementation measures all other 31 world gradients for both `SHAM_MEASURE_FULL32` and `SUBSPACE_PROJECT_FULL32`. In the SHAM arm, after those measurements it also calls the full subspace projection routine, including target-norm scans, Gram construction, pseudoinverse, projected-gradient construction, and post-projection dot measurements, even though SHAM discards the projected gradient and applies the original source gradient.

That extra solve is not part of the scientific intervention assigned to SHAM. SHAM's required behavior is measurement-only trajectory neutrality: measure the same 31 target gradients at the identical pre-update state/RNG, restore the source RNG, apply the untouched source gradient, and match CONTROL exactly.

## Candidate repair

The candidate changes only the SHAM branch inside `_measured_step`:

- keep all 31 target-gradient backward measurements unchanged;
- keep RNG reset/restoration unchanged;
- compute the same ordered raw source-target dot products needed for `negative_targets`;
- do not build the conflict subspace, Gram matrix, pseudoinverse, projected gradient, or post-projection dots for SHAM;
- apply `source_grads` exactly as before;
- leave PROJECT behavior byte-for-byte at the algorithmic level: it still calls the existing `_project_source` implementation and uses the same float64 Gram / `pinv(..., rtol=1e-12)` rule.

## Scientific invariants

This candidate does **not** change:

- A_FIXED architecture or 10,000,000 trainable parameter count;
- optimizer, LR, weight decay, clipping, tokenizer, objective, or dataset;
- 32-world population or source order;
- 32 exposures/world or effort cycle 1/2/4/8;
- target-gradient population (all other 31 worlds);
- source/target RNG geometry;
- SHAM applied gradient;
- PROJECT intervention;
- thresholds or reducer ordering;
- continuation/checkpoint/artifact chain semantics;
- authorization flags.

## Required equivalence gate before any repair execution

Before sealing a repair run, tests must establish that under the same reconstructed state and a bounded deterministic measured step:

1. old SHAM and repaired SHAM produce identical source loss, target-loss map, ordered raw-dot map, negative-target list, model state, optimizer state, and RNG state;
2. repaired SHAM still equals CONTROL at the existing per-chunk exact-state gate;
3. PROJECT outputs are unchanged for the same inputs;
4. no new scientific authorization is inferred from the optimization itself.

No EXP-302, EXP-320, scale, 30M, or 100M authorization is granted by this candidate.
