# EXP-327 — Iterative Minimal Failing Subset Localization

EXP-326 sealed `QUARTET_LEVEL_BREAK_PRESENT`: every registered pair passed, while iterative `Q0123` and the iterative octet failed. EXP-326 did not enumerate cross-pairs inside worlds 0..3.

EXP-327 freezes the complete non-singleton subset lattice of iterative worlds 0,1,2,3 below and including the quartet: all 6 pairs, all 4 triples, and Q0123. Every group is an independent clone from the same immutable A_FIXED step-2048 model/AdamW/RNG state.

Each world receives 32 exposures with the same world-local effort trace `1,2,4,8` repeated eight times. Checkpoints are 8, 16, and 32 exposures/world. A group passes only when every member world simultaneously reaches TF token accuracy >=0.99 and TF full-answer exact >=0.90.

The exact Q0123 geometry is repeated. If it passes, the court returns `PARENT_QUARTET_REPRODUCTION_MISMATCH` and makes no smaller-subset causal claim. Otherwise the court checks subset monotonicity and localizes the earliest failing cardinality: pair, triple, or quartet-only.

No architecture, tokenizer, objective, optimizer family, LR, exposure, effort, or scale change is permitted. EXP-302, EXP-320, scale, 30M and 100M authorization remain false.
