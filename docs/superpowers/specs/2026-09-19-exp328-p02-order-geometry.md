# EXP-328 — P02 Order / Retention Geometry Court

EXP-327 sealed `PAIR_MINIMAL_FAILURE_PRESENT` under the frozen A_FIXED 10M Stage-A court. Of all six iterative pairs inside worlds 0..3, only `P02` failed. The only failing triples were `T012` and `T023`, exactly the registered supersets containing P02, with no monotonicity violation.

EXP-328 does **not** change architecture, tokenizer, objective, optimizer, LR, exposure budget, effort cycle or scale. It asks a narrower causal question: does the P02 failure depend on update ordering / retention geometry?

Every arm starts from an independent clone of the same immutable step-2048 model + AdamW + RNG state. Worlds 0 and 2 each receive exactly 32 updates and the same world-local effort trace `1,2,4,8` repeated eight times.

Registered arms:

- `ALT_0_2`: exact EXP-327 P02 schedule; for each local exposure update world 0 then world 2. This is the mandatory parent-reproduction arm.
- `ALT_2_0`: reverse within-exposure direction.
- `BLOCK_0_2`: train world 0 for all 32 exposures, then world 2 for all 32.
- `BLOCK_2_0`: reverse block order.

Primary evaluation occurs only after both worlds have received exactly 32 exposures. An arm passes only if **both** worlds simultaneously satisfy TF token accuracy >= 0.99 and TF full-answer exact >= 0.90.

Decision order:

1. malformed/nonfinite/incomplete evidence → `INVALID_ORDER_GEOMETRY_COURT`;
2. if exact `ALT_0_2` reproduction passes → `PARENT_P02_REPRODUCTION_MISMATCH`;
3. if reproduction fails and any of the other three arms passes → `ORDER_GEOMETRY_SENSITIVE_INTERFERENCE`;
4. otherwise → `ORDER_GEOMETRY_INVARIANT_PAIR_FAILURE`.

A sensitive result localizes causality only to the preregistered schedule geometry. An invariant result rules out only these simple reorderings; it does not prove gradient conflict, representational collision or a universal forgetting mechanism.

EXP-302, EXP-320, scale, 30M and 100M authorization remain false.
