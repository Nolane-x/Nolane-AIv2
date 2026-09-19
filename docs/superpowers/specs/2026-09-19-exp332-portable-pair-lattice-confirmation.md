# EXP-332 — Portable Pair-Lattice Projection Confirmation

EXP-331/EXP-331R observed all six PROJECT pair arms passing with no regressions, but their historical parent gate failed because it demanded bitwise equality of model and optimizer state with the original EXP-327 hosted-run artifact.

That gate has now been empirically shown to be non-portable. The exact immutable EXP-327 workflow was rerun as attempt 2 of run `35413434081` without changing source, checkpoint, workflow, Python, PyTorch, training geometry, or preregistration. Attempt 2 preserved the EXP-327 decision, P02-only failure, every final pair metric, every pair pass/fail result, and every RNG digest, while all six pair model/optimizer digests differed from attempt 1. The attempt-2 pair state digests exactly matched EXP-331R CONTROL states.

EXP-332 preregisters the correction before another scientific run. Historical cross-run reproduction is defined by immutable provenance plus exact behavioral metrics/pass vector and RNG. Bitwise model/optimizer equality is not required across historical hosted-run boundaries because the immutable replay demonstrated that it is not reproducible. Within the new run, however, SHAM must still match CONTROL bit-for-bit in model, optimizer, RNG and primary metrics, preserving the strongest causal instrumentation gate.

The scientific arms, model, optimizer, learning rate, clipping, exposure budget, effort cycle, pair lattice and EXP-330 projection rule are unchanged. EXP-332 reruns all six CONTROL/SHAM/PROJECT pair courts from the same immutable 10M reconstruction.

The strongest positive disposition remains `PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION`: P02 must be rescued and all five previously passing pair controls must remain passing under PROJECT.

This court does not test triples, quartets or octets. It does not authorize EXP-302, EXP-320, scaling, 30M or 100M.
