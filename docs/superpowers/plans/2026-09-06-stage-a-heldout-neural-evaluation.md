# Stage-A Held-out Neural Evaluation Implementation Plan

1. Domain-separate curriculum generation between `augmentation` and `evaluation` RNG streams.
2. Add non-mutating neural evaluator for belief, conflict, and semantic-fidelity heads.
3. Retain aggregate metrics and raw per-batch metrics for future paired statistics.
4. Run identical held-out batches before and after bounded neural training.
5. Bind pre/post model digests and every training/evaluation batch digest into a self-hashed report.
6. Hard-code `EV-E2 / UNVERIFIED` and reject stream aliasing, promotion, batch-lineage drift, and report tampering.
7. Expose a CLI and clean-CI smoke path.
8. Preserve frozen Stage-A protocol bytes unchanged.
