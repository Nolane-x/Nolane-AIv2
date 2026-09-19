# EXP-331 — Complete Pair-Lattice Projection Safety/Efficacy Court

EXP-330 sealed `CONFLICT_PROJECTION_RESCUE` on the unique failing iterative pair P02. EXP-331 asks whether the exact registered projection mechanism rescues that parent failure without introducing regressions anywhere else in the complete six-pair lattice over iterative worlds 0..3.

The court remains A_FIXED at exactly 10M trainable parameters and starts every pair/mode arm from the same immutable step-2048 reconstruction. No architecture, tokenizer, objective, optimizer, learning rate, exposure budget, effort cycle or scale changes are permitted.

For each pair `P01/P02/P03/P12/P13/P23`, three independent arms are run:

1. `CONTROL` — exact EXP-327 pair schedule, 32 exposures/world, ascending within-pair order, world-local effort cycle 1,2,4,8.
2. `SHAM` — same measured-update machinery as EXP-330, including partner-gradient measurement and RNG restoration, but no projection. Its final primary metrics and model/optimizer/RNG digests must exactly match CONTROL.
3. `PROJECT` — the exact EXP-330 rule: if the same-state source/partner gradient dot product is negative and partner norm-squared exceeds `1e-24`, remove the source component along the partner gradient before the inherited global clip and AdamW step.

The CONTROL lattice must reproduce the authoritative EXP-327 final pair metrics and state digests exactly, not merely the parent pass/fail vector. This locks P02 as the only failing control pair and the other five pairs as passing controls.

The strongest positive disposition is `PAIR_LATTICE_PROJECTION_RESCUE_NO_REGRESSION`: P02 projection must activate and pass, while all five previously passing pairs must remain passing under PROJECT. A P02 rescue that breaks any previously passing pair is separately classified and cannot count as the strongest result.

This court does not test triples, quartets or octets and does not authorize EXP-302, EXP-320, scaling, 30M or 100M.
