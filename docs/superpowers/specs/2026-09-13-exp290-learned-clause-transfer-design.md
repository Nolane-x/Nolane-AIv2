# EXP-290 Learned Clause Transfer Court — Design

## 1. Scientific question

EXP-289 established scoped EV-E3 evidence that exact episode-local nogoods can eliminate repeated dead-end work without over-pruning on the frozen synthetic family. EXP-290 asks the next narrower question from the V0.16.1 program:

> Can a learned surface-invariant correspondence mechanism transfer clauses learned from one episode into a structurally isomorphic but surface-randomized episode, recovering a material fraction of oracle structural-transfer value without violating soundness or the protected solution-rate floor?

This is a DEVELOPMENT / EV-E2 court first. It does not claim lifelong lemma economy, unrestricted cross-domain clause reuse, semantic authority, integrated NLM success, or scientific promotion. EXP-291 remains blocked unless this court closes recurrently positive.

Base authority: `main@e7a034365e542895b772701a247f6ad9b931bfc9`.

Frozen Stage-A protocol digest remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`; EXP-290 does not modify Stage-A protocol bytes.

## 2. Why a same-weights four-mode transfer court

The decisive uncertainty is structural transfer, not whether local nogoods are useful. Training separate end-to-end agents would mix clause-transfer quality with optimizer variance, search-policy drift, representation drift and local-memory behavior.

Each canonical root therefore trains exactly one correspondence model and freezes it. The identical frozen weights are then evaluated on the same source/target pairs under four modes:

1. `NULL_TRANSFER_CONTROL` — the correspondence model is executed and charged, but no source clause is made available to the target search.
2. `RAW_SURFACE_TRANSFER_CONTROL` — source clause keys are copied literally into the target library. Source and target surface namespaces are disjoint by construction, so literal key reuse cannot masquerade as structural transfer.
3. `LEARNED_STRUCTURAL_TRANSFER` — source clause variables are mapped into target variables only by the frozen learned correspondence model. The resulting transferred clauses are queried with exact symbolic subset semantics.
4. `ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND` — the same correspondence model computation is still executed and charged, but evaluator-only source→target structural correspondence is used to map source clauses exactly.

The source phase is identical for all four target modes and is executed once per paired example. The target search algorithm, restart schedule, episode ceiling and clause-query semantics are identical across modes. Only the source-clause transfer map differs.

`ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND` is explicitly non-deployable. Evaluator correspondence truth may enter only the oracle mode and post-hoc diagnostics. It may not train, calibrate, threshold, select or otherwise alter `LEARNED_STRUCTURAL_TRANSFER` on held-out evaluation pairs.

## 3. Transfer world geometry

EXP-290 uses deterministic paired binary-CSP worlds with a hidden structural template and two independent surface realizations.

Per pair:

- `variables = 8`
- binary domains
- source and target use disjoint opaque surface variable names
- source and target are structurally isomorphic under one hidden bijection
- each world contains pairwise forbidden assignments that create two-literal dead ends
- at least one verified global solution exists
- `restarts = 4`
- `max_search_steps = 24`
- `timesteps = 4`
- `d_model = 64`
- `hidden_size = 48`
- `target_parameters = 500000`
- `batch_size = 8`
- `noise_std = 0.05`
- optimizer = AdamW
- `lr = 0.002`
- `weight_decay = 0.0`

The hidden source→target bijection, canonical structural template and target validity truth are evaluator-only metadata.

Observable variable states carry noisy structural signatures generated from the variable's local constraint role. Surface names and tensor order are randomized independently between source and target. The correspondence model receives only observable source/target variable states and shared event context; it never receives hidden canonical role IDs, evaluator mapping IDs, solution truth, future search outcomes, or target clause truth at evaluation time.

## 4. Learned correspondence model

The model embeds each source and target variable into a shared normalized representation:

`z_i = normalize(MLP(variable_state_i + recurrent_event_context))`.

For a source variable `i` and target variable `j`:

`similarity(i,j) = dot(z_source_i, z_target_j)`.

The learned mapping is deterministic row-wise top-1 target selection with ties broken by ascending target tensor index. A transferred source clause is rejected as `mapping_collision` if two distinct source literals map to the same target variable. Collision rejection is determined from model outputs alone and does not consult evaluator truth.

There is no threshold sweep, temperature calibration, Hungarian/oracle assignment, beam search, retry with a second candidate, or post-result correspondence repair in V1.

The model is executed for every mode. Parameter count, active functional parameters, optimizer-visible parameters and analytical neural inference cost therefore match exactly across modes.

## 5. Training contract

Training uses only fresh EXP-290 `augmentation` source/target pairs. Evaluation pairs are unopened and disjoint until the model state is frozen.

For augmentation pairs only, generator correspondence labels supervise a fixed bidirectional correspondence objective:

- source→target row cross-entropy
- target→source row cross-entropy

Total loss:

`loss = 0.5 * CE(source_to_target) + 0.5 * CE(target_to_source)`

There are no auxiliary search losses, class weights, focal terms, contrastive margin sweeps, temperature tuning, curriculum switches, early stopping, hard-negative mining, replay, evaluator-guided filtering, or post-result loss changes.

Generator correspondence labels are allowed during DEVELOPMENT training because EXP-290 explicitly asks whether surface-invariant structural correspondence can be learned from synthetic aligned pairs. Held-out evaluation correspondence labels remain evaluator-only.

## 6. Source clause acquisition

Source clauses are not injected from evaluator truth.

Each source episode runs a frozen deterministic bounded search. A source clause may be stored only when:

1. the source search actually reaches the partial assignment;
2. the same arm-observable propagation/search semantics declare it a dead end;
3. the assignment is non-empty and contains exactly two literals under this frozen family;
4. no hidden target mapping, source solution enumeration, oracle conflict core, or future-path truth gates insertion.

The stored source key is the exact two-literal surface assignment reached by the source search. Post-hoc evaluator truth may audit whether it is a valid structural nogood but must not alter source behavior.

Source search and source clause acquisition are shared across all four target modes. A mode cannot selectively alter which source clauses exist.

## 7. Target transfer semantics

The target phase starts with a fresh episode and no target-local learned clauses. Target-local insertion is disabled in V1 so local learning cannot obscure the causal contribution of transferred source clauses.

All modes run the same deterministic target search and exact subset-match query semantics.

### `NULL_TRANSFER_CONTROL`

No source clause enters the target library. The correspondence network is still executed and charged.

### `RAW_SURFACE_TRANSFER_CONTROL`

Literal source surface keys are copied without renaming. Generator validation requires source/target surface namespaces to be disjoint, so a raw source variable name cannot equal a target variable name. Any raw-surface hit is an integrity failure.

### `LEARNED_STRUCTURAL_TRANSFER`

Each source clause variable is mapped to its row-wise top-1 target variable using only frozen learned embeddings. Values are preserved because source and target domains share the same binary value semantics in this V1 court. If the two source literals collide onto one target variable, the clause is rejected and recorded; no evaluator-assisted fallback is permitted.

Accepted learned clauses enter an exact target-side symbolic store and prune only when the transferred clause is a subset of the current target partial assignment.

### `ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND`

Evaluator-only hidden bijection maps the exact source clause into target surface variables. The resulting exact target clause uses the same symbolic store/query implementation as learned mode.

## 8. Frozen transfer opportunity manifest

The primary denominator may not depend on a target mode's own trajectory.

For each held-out source/target pair, the generator creates an evaluator-only transfer opportunity manifest before any mode executes. It contains target dead-end states that are structurally equivalent to source-observed clauses under the hidden bijection and that a deterministic reference target traversal would encounter without transferred memory.

Each manifest item binds:

- source clause receipt digest
- hidden mapped target clause digest
- target restart index
- target reference dead-end state digest
- pair lineage digest

The manifest is never delivered to learned or raw control paths.

Define:

`structural_repeat_dead_end_rate = target_structural_reentries / predeclared_transfer_opportunities`.

The denominator is evaluator-frozen. A mode can lower only the numerator by preventing an eligible target dead-end reentry before full dead-end cost.

Pairs with zero opportunities remain in raw evidence and safety/cost summaries but are excluded from this rate denominator with an explicit `transfer_rate_defined=false` flag.

## 9. Soundness and transfer diagnostics

Every target prune event is audited post hoc against exact target truth. Evaluator truth is never fed back into search.

Required diagnostics include:

- source→target variable correspondence accuracy
- target→source variable correspondence accuracy
- exact transferred-clause recovery rate
- learned mapping collision rate
- accepted transferred clauses
- transferred-clause query hits
- prevented structural repeat count
- structural repeat dead-end rate
- invalid transferred-clause prune count
- valid-state over-prune rate
- source clause soundness violations
- target transferred-clause soundness violations
- raw-surface transfer hit count
- verified-solution rate
- censored episode count

`valid_state_overprune_rate = invalid_transferred_prune_events / max(transferred_prune_events, 1)`.

A nonzero soundness or over-prune result is valid negative experimental evidence. Validators must preserve it rather than deleting the row as infrastructure-invalid unless the receipt itself is inconsistent.

## 10. Economic metric and oracle-value capture

All four target modes use the same maximum episode budget. Analytical accounting charges:

- shared source search work
- source clause canonicalization/storage
- correspondence-model inference in every mode
- learned mapping operations when executed
- target clause storage/query comparisons when executed
- target search work
- verifier work

No hardware-profiler FLOP claim is made.

For each canonical root let target mean accounted cost be:

- `C0` = `NULL_TRANSFER_CONTROL`
- `CR` = `RAW_SURFACE_TRANSFER_CONTROL`
- `CL` = `LEARNED_STRUCTURAL_TRANSFER`
- `CO` = `ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND`

Define:

`H_oracle = C0 - CO`

`H_learned = C0 - CL`

and, only when `H_oracle > 0`:

`capture = H_learned / H_oracle`.

Capture is not clipped. A negative learned benefit stays negative; learned better than oracle may exceed one and remains visible for audit.

`CR` is a surface-only negative control, not part of the capture denominator.

## 11. Frozen DEVELOPMENT geometry

Four canonical roots are all decision roots:

- canonical indices: `0,1,2,3`
- root prefix: `20260913-exp290-learned-clause-transfer-v1-dev`
- `train_replicates = 64` per root
- `eval_replicates = 32` per root
- `eval_start_replicate = 10000`
- batch size `8`
- 512 training pairs/root
- 256 held-out pairs/root
- 2,048 training pairs total
- 1,024 held-out pairs total

Per canonical root, model initialization, augmentation and evaluation seeds are independently derived using the repository seed-domain machinery under namespace `EXP-290`.

Training and evaluation replicate namespaces are disjoint. No EXP-289 confirmatory observation and no EXP-287 DEVELOPMENT observation is reused for fitting or selection.

## 12. Root classification

A root is `LEARNED_CLAUSE_TRANSFER_ESTABLISHED` only if all predicates hold:

1. fresh oracle structural transfer reproduces positive economic headroom: `H_oracle > 0`;
2. oracle mode reduces structural repeat dead-end rate versus null by at least `0.25` relative;
3. learned transfer produces positive economic headroom: `H_learned > 0`;
4. learned oracle-value capture is at least `0.50`;
5. learned source→target variable correspondence accuracy is strictly greater than `0.50`;
6. learned exact transferred-clause recovery rate is strictly greater than `0.50`;
7. learned structural repeat dead-end rate is strictly lower than null structural repeat dead-end rate;
8. learned verified-solution rate is at least null verified-solution rate minus `0.01`;
9. learned valid-state over-prune rate is at most `0.005`;
10. learned evaluation receives no oracle correspondence, evaluator validity truth or target clause truth;
11. raw-surface transfer hit count is exactly zero and source/target surface namespaces are disjoint.

If predicate 1 or 2 fails, classification is `ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED` because the fresh root does not reproduce the transfer value required to judge the learned mechanism.

If oracle transfer headroom exists but any predicate 3–11 fails, classification is `LEARNED_CLAUSE_TRANSFER_NOT_ESTABLISHED`.

The `0.25` oracle repeat-rate reduction, `0.50` capture, `>0.50` correspondence and exact-clause recovery thresholds, `0.005` over-prune ceiling and `-0.01` solution-rate floor are fixed before held-out visibility.

## 13. Cross-root decision

All four roots are decision roots; no root selection is allowed.

- `CLAUSE_TRANSFER_RECURRENT` only if all four roots are `LEARNED_CLAUSE_TRANSFER_ESTABLISHED`.
- `CLAUSE_TRANSFER_INTERMITTENT` if at least one but not all four roots establish learned transfer and no root lacks oracle headroom.
- `CLAUSE_TRANSFER_NOT_ESTABLISHED` if zero roots establish learned transfer and all four reproduce oracle transfer headroom.
- `ORACLE_TRANSFER_REPLICATION_INCOMPLETE` if any root is `ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED`.

Only `CLAUSE_TRANSFER_RECURRENT` may set:

- `successor_design_authorized = true`
- `authorization_scope = DESIGN_EXP291_SPURIOUS_COUNTEREXAMPLE_COURT_ONLY`

Every other result sets `successor_design_authorized=false` and `authorization_scope=NONE`.

No EXP-290 outcome directly authorizes implementation of EXP-291, confirmatory data, challenge materialization, semantic authority, lifelong clause reuse, Lemma Economy or integrated-system promotion.

## 14. Evidence boundary

EXP-290 V1 is `EV-E2 / DEVELOPMENT` regardless of outcome.

Every root and cross receipt must record:

- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_correspondence_used_for_training = false`
- `evaluation_correspondence_used_by_learned_mode = false`
- `evaluator_truth_used_by_learned_control = false`
- `oracle_mode_deployable = false`
- `target_local_clause_learning_enabled = false`
- `lifelong_clause_reuse_claimed = false`
- `semantic_authority_claimed = false`

## 15. Integrity and serialization

Receipts use canonical JSON with string keys from construction time. Every artifact contains `receipt.json`, matching SHA-256 sidecar, protocol digest, geometry digest, source-tree `code_digest`, exact root identity and enough primitive rows to recompute every decision predicate.

Validators must reject at least:

- forged favorable classifications after rehashing
- protocol/geometry/code identity mismatch
- duplicate or missing canonical roots
- model-state drift across evaluation modes
- training/evaluation lineage overlap
- source/target namespace overlap
- hidden evaluator correspondence entering learned mode
- target validity truth entering control
- target-local clause insertion
- raw-surface hit count above zero
- transfer-opportunity denominator drift
- deleted/reordered primitive rows
- learned mapping collision silently repaired with oracle truth
- invalid prune events omitted from safety metrics
- root or cross decision labels inconsistent with primitive fields
- canonical serialize→parse→serialize drift

The learned model digest is captured after training and must remain byte-identical across all four evaluation modes.

## 16. Release and one-shot execution discipline

Before any held-out DEVELOPMENT pair is opened:

1. design and implementation plan are committed;
2. implementation proceeds through test-first RED/GREEN seams;
3. dedicated EXP-290 contract CI, generic core CI and model-smoke are GREEN on one exact pre-data head;
4. freeze guard proves zero prior EXP-290 DEVELOPMENT runs;
5. a marker-only commit adds `protocols/exp290-learned-clause-transfer-release.lock` and changes no scientific code/test/workflow semantics;
6. the marker triggers exactly one authoritative DEVELOPMENT workflow;
7. duplicate/rerun attempts fail before root data execution;
8. four root receipts reduce into one sealed cross receipt;
9. artifacts are independently audited before PR closure.

After held-out result visibility, V1 scientific semantics are immutable. An unfavorable valid result is not an instrumentation defect.

## 17. No post-result tuning

After any held-out EXP-290 result is visible, V1 forbids:

- changing correspondence selection semantics
- changing any success threshold
- changing loss weights
- adding calibration or a second-choice mapping fallback
- increasing sample counts
- changing surface-randomization strength
- changing source/target clause geometry
- adding target-local learning
- dropping a canonical root
- replacing a failed root with a new seed
- consuming confirmatory/challenge data

A true instrumentation defect may only be repaired through a separately versioned execution attempt that documents the invalid attempt and preserves scientific semantics.

## 18. Interpretation boundary

`CLAUSE_TRANSFER_RECURRENT` would mean only that, on this frozen synthetic isomorphism family, a supervised learned correspondence mechanism recurrently transfers source dead-end clauses across opaque surface randomization, recovers a majority of fresh oracle transfer value, and preserves the protected safety/solution floors. It would justify designing EXP-291.

Any weaker result blocks EXP-291 under this dependency hypothesis. It does not erase EXP-289's positive episode-local nogood result. It localizes the bottleneck to structural clause transfer under the frozen EXP-290 representation/objective.
