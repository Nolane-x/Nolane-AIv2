# EXP-330 — P02 Conflicting-Gradient Projection Rescue Court

EXP-329 sealed `BIDIRECTIONAL_LOCAL_CROSS_DAMAGE` on P02. At registered later checkpoints, source updates improved their own answer-only loss while increasing the other world's loss, and the source/target gradient dot product was negative. EXP-330 tests a narrow causal intervention without changing scale or architecture.

All arms start from independent copies of the exact immutable step-2048 A_FIXED 10M model, AdamW state and torch CPU RNG. All receive exactly 32 exposures of world 0 and 32 of world 2 in the same per-round order `0 -> 2`, with world-local effort cycle `1,2,4,8`, LR `5e-5`, weight decay `0.01`, and global gradient clipping at `1.0`.

Registered arms:

1. **CONTROL_ALT** — exact inherited ALT_0_2 training primitive.
2. **SHAM_MEASURE_ALT** — before every source step, measure source and target gradients from the same pre-step state. Restore the exact source gradient and the RNG state immediately after the source backward, apply no projection, then clip and AdamW-step. Its final model/optimizer/RNG digests and primary metrics must exactly match CONTROL_ALT. Otherwise the entire comparison is `SHAM_TRAJECTORY_MISMATCH`.
3. **PROJECT_CONFLICT_ALT** — identical measured-step machinery, but when `dot(g_source,g_target) < 0` and target norm-squared exceeds `1e-24`, replace the source gradient with
   `g_source - dot(g_source,g_target)/||g_target||^2 * g_target`
   before the same global clip and AdamW step.

The auxiliary target-gradient calculation never calls optimizer.step and its RNG consumption is removed. Thus the authoritative RNG progression for a measured source step is the source-forward/backward progression only.

Primary final fit uses the inherited simultaneous floor: each world must independently reach teacher-forced answer-token accuracy >= 0.99 and full-answer exact >= 0.90.

Decision order is fail-closed: malformed/nonfinite evidence; parent reproduction mismatch; sham trajectory mismatch; projection not triggered; projection rescue; projection no-rescue.

A rescue result is scoped: under this frozen P02 trajectory, removal of the currently opposing target-gradient component is sufficient for simultaneous fit. It is not a claim that gradient projection is universally optimal. A no-rescue result means this registered orthogonal removal alone is insufficient.

EXP-302, EXP-320, scale, 30M and 100M authorization remain false.
