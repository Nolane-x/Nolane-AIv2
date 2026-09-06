# Stage-A Multi-seed Held-out Implementation Plan

1. Retain one immutable outcome record per requested model seed.
2. Bind each successful outcome to its held-out evidence artifact and report digest.
3. Retain failed seed lineage explicitly and invalidate all aggregate mean/CI statistics if any requested seed fails.
4. Compute deterministic multi-seed uncertainty and sign-consistency diagnostics only when every requested seed succeeds.
5. Reject seed dropping, reordering, promotion, invalid analysis RNG, or digest tampering.
6. Expose a CLI and CPU-safe clean-CI smoke run.
7. Preserve the frozen Stage-A protocol bytes unchanged.
