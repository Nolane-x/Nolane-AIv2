# EXP-329 — P02 Direct Cross-Update Interference Court

EXP-328 sealed `ORDER_GEOMETRY_INVARIANT_PAIR_FAILURE`: the exact P02 alternating failure reproduced, reversing alternating order did not rescue it, and neither block ordering achieved simultaneous fit. Block schedules retained only the world trained in the last block, while both alternating schedules ended with world 2 fit and world 0 failed.

EXP-329 is an observational/local-interaction court. It does not change model architecture, tokenizer, objective, optimizer, learning rate, exposure budget, effort cycle, or scale. The authoritative main trajectory is the exact `ALT_0_2` replay from the same immutable step-2048 A_FIXED state.

Before rounds 0, 1, 2, 4, 8, 16, 24, and 31, the court freezes the current replay state and performs non-authoritative probes:

1. compute answer-only gradients for worlds 0 and 2 from the identical pre-round model state at the registered next-round effort;
2. record gradient norms, dot product, and cosine similarity;
3. independently deep-clone the current model+optimizer for each directed one-step probe;
4. update only the source world once at the registered effort;
5. evaluate source and target answer-only losses at that same effort;
6. restore the main torch CPU RNG so probes cannot perturb the authoritative replay.

A directed local cross-damage event is registered only when the source update improves its own loss by at least `1e-6` and increases the other world's loss by at least `1e-6`. Gradient cosine is evidence only and cannot independently determine the disposition.

After probes, the main replay continues unchanged with world 0 then world 2. After 32 exposures/world, the exact parent simultaneous-fit floor is checked. If the parent failure does not reproduce, no local-interaction claim is permitted.

Possible bounded dispositions are invalid/reproduction mismatch, bidirectional local cross-damage, world-0-only local cross-damage, world-2-only local cross-damage, or no registered local cross-damage.

All EXP-302, EXP-320, scale, 30M, and 100M authorization remains false.
