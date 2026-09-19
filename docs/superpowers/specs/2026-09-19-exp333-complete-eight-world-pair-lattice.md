# EXP-333 — Complete Eight-World Pair-Lattice Projection Court

EXP-332 established, on the frozen A_FIXED 10M reconstruction and the six pair subsets of iterative-grid-and-maze worlds 0..3, that the exact EXP-330 conflicting-gradient projection rule rescues the registered P02 failure without a registered pair regression.

EXP-333 tests whether that bounded result generalizes across the complete pair lattice of the same eight-world family. Before execution, the population is fixed to worlds 0..7 and all 28 unordered pairs. Every pair receives independent CONTROL, SHAM and PROJECT clones from the same immutable reconstruction, for 84 total arms.

No scientific geometry changes: AdamW state, LR 5e-5, weight decay 0.01, clip 1.0, 32 exposures per world, effort cycle 1/2/4/8, 64 optimizer updates per pair arm, and the exact EXP-330/EXP-332 projection condition are preserved.

The six EXP-332 pairs are historical behavioral anchors. Their CONTROL arms must reproduce pair membership, token metrics, full-answer exact metrics, RNG digest and pass/fail status. As established by the immutable EXP-327 replay, cross-run model/optimizer bitwise equality is not a portable historical criterion. Within EXP-333, every SHAM must still equal its CONTROL exactly in metrics, model digest, optimizer digest and RNG digest.

All CONTROL failures among the 28 pairs are determined by the preregistered court; no failing pair may be selected after seeing results. The reducer then classifies rescued failures, unresolved failures, PROJECT regressions, and baseline failures for which projection never triggered.

The strongest positive disposition, `COMPLETE_8WORLD_PAIR_LATTICE_RESCUE_NO_REGRESSION`, requires at least one CONTROL baseline failure, projection activation for every baseline failure, rescue of every baseline failure, and zero regression among CONTROL-passing pairs.

EXP-333 does not test triples, quartets, larger subsets, or the eight worlds jointly. It does not authorize EXP-302, EXP-320, scaling, 30M or 100M.
