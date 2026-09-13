# EXP-290 Structural Clause Transfer Court — Design

## 1. Research question

EXP-289 established scoped EV-E3 evidence that exact episode-local nogoods can reduce repeated dead ends while preserving the frozen safety floors. EXP-290 asks the next narrower question:

> Can a learned transfer mechanism reuse an observed source nogood on a held-out isomorphic target whose variable identity and surface ordering have been independently randomized, recovering a material fraction of oracle source→target transfer value without unsafe pruning?

This is a new DEVELOPMENT / EV-E2 court. It is not part of the frozen Stage-A V1 experiment list and does not modify `protocols/stage_a_v1.json` or its digest. A positive result may authorize only a separately preregistered EXP-291 design; it does not authorize scientific promotion, open-domain clause transfer, arbitrary symbol remapping, or lemma generation.

Base authority: `main@e7a034365e542895b772701a247f6ad9b931bfc9`.

Frozen Stage-A protocol lineage digest remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

## 2. Why EXP-290 must isolate transfer rather than re-test local memory

EXP-289 already answered whether exact same-episode nogoods can be useful. Replacing its local-memory control with a no-memory baseline would allow EXP-290 to win by rediscovering the parent result rather than transferring knowledge.

Therefore every EXP-290 target arm retains the same exact episode-local target nogood mechanism. The only intervention is whether source-observed nogoods are available on the target before that target has encountered the equivalent dead end.

The three target modes are:

1. `LOCAL_ONLY_CONTROL` — source clauses are encoded and transfer scores are computed and charged, but no source-derived target clause is activated. The target still learns exact episode-local nogoods after its own observed contradictions.
2. `LEARNED_STRUCTURAL_TRANSFER` — each source-observed one-literal nogood is translated by the frozen learned transfer head into exactly one target literal; translated clauses are available before the corresponding target contradiction is observed. Target-local memory remains active.
3. `ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND` — the learned transfer head is still executed and charged, but the generator-hidden source→target variable bijection replaces its translation. This mode is non-deployable and exists only to establish fresh transfer headroom.

One trained model is evaluated in all three modes. Model state is byte-identical across modes.

## 3. Scope of V1 surface randomization

V1 tests structural transfer under a deliberately bounded surface transformation. Source and target share one latent binary CSP and the same value alphabet `{0,1}`, but independently randomize:

- visible variable identity;
- visible variable order;
- restart variable ordering;
- event ordering;
- nuisance surface noise.

A non-identity variable permutation is mandatory for every evaluated source/target pair.

V1 does **not** randomize the value alphabet. Passing therefore does not establish arbitrary token/value renaming, semantic paraphrase invariance, cross-domain transfer, or robustness to bad encodings. Those claims remain outside scope; encoding/counterexample reliability belongs to EXP-291 and later seams.

## 4. Paired source/target world

Each paired world contains one latent problem and two independently surfaced views.

The latent problem preserves the EXP-289 bounded synthetic family:

- `variables = 8`;
- binary values;
- exactly `3` decoy variables;
- `restarts = 4`;
- public contradiction semantics equivalent to the EXP-289 unary target constraints;
- source and target are globally solvable;
- repeated-dead-end opportunities are frozen before any arm action.

For every pair the generator samples a target variable permutation `p` with `p != identity`. If source visible variable `s_i` denotes latent variable `i`, target visible variable `t_j` denotes latent variable `p^{-1}(j)`. The hidden mapping is serialized only in evaluator metadata.

The learned mode may observe source and target surface tensors and the source-observed literal itself. It may not observe `p`, target-equivalent labels, valid-solution metadata, evaluator truth, future target contradictions, or oracle receipts.

## 5. Source phase is common and causal

Before target-mode execution, one common source phase runs once per paired episode using exact episode-local nogood semantics inherited from EXP-289.

A transferable source clause exists only if:

- the source partial assignment is actually reached;
- a public contradiction is observed before insertion;
- the source clause is non-empty;
- no evaluator truth gates insertion.

V1 transfers the resulting one-literal source nogoods produced by this frozen synthetic family. The source clause set is sealed before any target mode runs and is identical for control, learned, and oracle modes.

The evaluator may then use the hidden isomorphism to construct a target transfer-opportunity manifest. That manifest is never delivered to `LEARNED_STRUCTURAL_TRANSFER` and cannot change target actions.

## 6. Learned transfer model

EXP-290 adds a transfer head to a matched 500k-parameter reasoner while retaining the existing event/variable/recurrent substrate.

Per root:

- `d_model = 64`
- `hidden_size = 48`
- `target_parameters = 500000`
- `timesteps = 4`
- `variables = 8`
- `decoys = 3`
- `restarts = 4`
- `max_search_steps = 24`
- `batch_size = 8`
- `noise_std = 0.05`
- optimizer = AdamW
- `lr = 0.002`
- `weight_decay = 0.0`

For one source literal `(source_variable, value)`, the transfer head receives:

- the source variable state at that visible source index;
- the preserved binary literal value;
- all target variable states.

It produces one score per target variable. `LEARNED_STRUCTURAL_TRANSFER` selects exactly one target variable using deterministic top-1 with ascending target-index tie break and preserves the source literal value. There is no probability threshold, calibration step, abstention threshold, top-k sweep, or cardinality sweep in V1.

The target transfer clause therefore has exactly one literal. Target execution uses exact subset matching after translation; approximate matching is confined to the learned source→target translation itself.

## 7. Matched compute and memory accounting

All three modes execute and charge the learned transfer scorer for every source clause.

Transferred clauses are represented as fixed source-clause slots rather than a deduplicating set. A duplicate learned prediction cannot reduce accounted query cost by shrinking memory. Each target query charges the same number of transferred-slot comparisons across all modes.

Target-local memory insertion, canonicalization, and comparison operations remain charged exactly as in EXP-289. Neural computation, source-clause encoding, transfer scoring, transferred-slot comparison, and local-memory operations all appear in the receipt.

A common per-episode cost ceiling is frozen before evaluation. Censored episodes remain scientific/development outcomes and are charged at that ceiling.

## 8. Training contract

Each canonical root trains exactly one model using only fresh EXP-290 `augmentation` source/target pairs. Evaluation pairs are disjoint and unopened until the model and all decision constants are frozen.

Training supervision may use the augmentation-only hidden source→target variable mapping because EXP-290 explicitly studies supervised structural transfer. Evaluation mappings may not train, calibrate, threshold, select, or otherwise change the learned mechanism.

The fixed training objective is:

`loss = branch_cross_entropy + verifier_binary_cross_entropy + transfer_mapping_cross_entropy`

All three coefficients are exactly `1.0`.

There are no class weights, focal terms, temperature tuning, curriculum switches, replay, hard-negative mining, early stopping, post-result loss-weight changes, or evaluation-informed selection.

## 9. DEVELOPMENT geometry

Four independent canonical roots reject one-seed transfer wins:

- canonical indices: `0,1,2,3`
- root prefix: `20260913-exp290-structural-clause-transfer-v1-dev`
- `train_replicates = 64` per root
- `eval_replicates = 32` per root
- `eval_start_replicate = 40000`
- batch size `8`
- training paired episodes/root: `512`
- held-out paired episodes/root: `256`
- total training pairs: `2,048`
- total held-out pairs: `1,024`

Model initialization, augmentation, and evaluation seeds are independently derived from repository seed-domain machinery under the `EXP-290` namespace. Source and target sub-seeds are deterministic domain-separated derivatives of the permitted augmentation/evaluation stream seed. No challenge seed, confirmatory beacon, EXP-289 confirmatory observation, or EXP-287 observation is reused for fitting or selection.

## 10. Primary transfer endpoint

RDER is not the EXP-290 primary endpoint because target-local memory can already prevent later repeats after the target has observed a contradiction. EXP-290 instead measures whether source knowledge prevents the target's **first source-equivalent dead-end encounter**.

For each paired episode, after the common source phase and before target arm execution, evaluator-only metadata freezes the set of target literals that are isomorphic equivalents of observed source nogoods.

Define:

`source_equivalent_target_dead_end_rate = reached_source_equivalent_target_dead_ends / predeclared_transfer_opportunities`

Lower is better. A transferred clause that prunes the equivalent target literal before contradiction prevents that opportunity. Opportunities with denominator zero remain in raw receipts but are excluded from this rate; no epsilon denominator rescue is allowed.

For each root let:

- `R0` = pooled rate for `LOCAL_ONLY_CONTROL`
- `RL` = pooled rate for `LEARNED_STRUCTURAL_TRANSFER`
- `RO` = pooled rate for `ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND`

Define fresh oracle transfer headroom:

`H_oracle = R0 - RO`

and learned transfer headroom:

`H_learned = R0 - RL`.

When `H_oracle > 0`, define transfer-value capture:

`capture = H_learned / H_oracle`.

Capture is not clipped.

## 11. Safety and diagnostics

Protected safety endpoints inherit EXP-289 semantics:

- learned valid-state over-prune rate `<= 0.005`;
- learned verified-solution rate `>= LOCAL_ONLY_CONTROL - 0.01`.

Additional held-out diagnostics are report-only and never tune the policy:

- exact source→target variable translation accuracy;
- non-identity permutation rate, required to be `1.0`;
- number of source clauses;
- number of learned transferred clauses used;
- number of oracle transferred clauses used;
- transferred prune count;
- target-local prune count;
- target RDER;
- accounted reasoning cost;
- learned/oracle/control cost ratios;
- duplicate transfer prediction count;
- source and target world digests.

Evaluator truth may label an observed prune as safe/unsafe only after the action is fixed.

## 12. Root classification

A root is `LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED` only if all predicates hold:

1. `H_oracle > 0` — fresh oracle source→target transfer headroom exists;
2. `H_learned > 0` — learned transfer prevents more source-equivalent first dead ends than local-only control;
3. `capture >= 0.50` — learned transfer recovers at least half of fresh oracle transfer headroom;
4. learned valid-state over-prune rate `<= 0.005`;
5. learned verified-solution rate is at least control rate minus `0.01`;
6. learned transferred prune count is greater than zero;
7. every held-out pair uses a non-identity source→target variable permutation;
8. evaluation mapping labels and oracle mapping are absent from learned action inputs.

If predicate 1 fails, root classification is `ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED`; the learned mechanism is not called negative because the fresh denominator for transfer value is absent.

If predicate 1 passes but any predicate 2–8 fails, root classification is `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED`.

The 0.50 capture threshold encodes a majority-of-oracle requirement. The safety floors are inherited unchanged from EXP-289 rather than relaxed for the learned mechanism.

## 13. Cross-root decision

All four canonical roots are decision roots.

- `STRUCTURAL_CLAUSE_TRANSFER_RECURRENT` only if all four roots are `LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED`.
- `STRUCTURAL_CLAUSE_TRANSFER_INTERMITTENT` if at least one but not all four roots establish transfer.
- `STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED` if zero roots establish transfer and all four reproduce positive oracle headroom.
- `ORACLE_TRANSFER_REPLICATION_INCOMPLETE` if any root is `ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED`.

Only `STRUCTURAL_CLAUSE_TRANSFER_RECURRENT` may set:

- `successor_design_authorized = true`
- `authorization_scope = DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY`

Every other disposition sets successor design authorization false.

No EXP-290 result directly authorizes EXP-291 implementation, confirmatory data, challenge materialization, Stage-A protocol modification, unrestricted learned-clause memory, lemma generation, or promotion.

## 14. Evidence boundary

Every EXP-290 V1 receipt must preserve:

- `evidence_level = EV-E2`
- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_mapping_used_for_training = false`
- `evaluation_mapping_used_by_learned_mode = false`
- `oracle_transfer_mode_deployable = false`
- `arbitrary_value_symbol_remapping_claimed = false`
- `cross_domain_transfer_claimed = false`
- `lemma_generation_claimed = false`.

## 15. Integrity and serialization

Root and cross receipts use canonical JSON and SHA-256 sidecars. They bind:

- exact repository head;
- source-tree code digest;
- Stage-A protocol digest;
- EXP-290 geometry digest;
- canonical root identity;
- model-state digest;
- source/target pair digests;
- source clause set digest;
- decision primitives and recomputed classification;
- evidence-boundary flags.

Validators reject forged decisions, duplicate roots, mixed code/geometry identities, training/evaluation overlap, identity permutations in held-out pairs, target-mode model drift, learned access to evaluator/oracle mapping, unsafe pre-action ground-truth gates, and canonical serialization drift.

## 16. One-shot release discipline

No EXP-290 held-out DEVELOPMENT target result may exist before all code/test/workflow gates are GREEN at an exact pre-marker head.

The release marker is a marker-only commit. It must bind the exact pre-marker head, frozen geometry, root set, protocol digest, and all decision constants. The DEVELOPMENT workflow accepts only the first authoritative run attempt for the PR and serializes duplicates so a second run cannot begin data execution.

After any held-out result is visible, V1 prohibits:

- changing surface randomization;
- adding value-label randomization;
- changing top-1 translation cardinality;
- adding thresholds or calibration;
- changing capture or safety thresholds;
- changing loss weights;
- increasing sample counts;
- dropping/replacing a canonical root;
- changing transfer-head geometry;
- rerunning an unfavorable valid result;
- consuming confirmatory/challenge data.

Instrumentation failure may be handled only through an explicitly versioned invalid-attempt record without changing scientific semantics. An unfavorable valid result is not an instrumentation failure.

## 17. Interpretation

`STRUCTURAL_CLAUSE_TRANSFER_RECURRENT` would mean only that, on this frozen binary synthetic family, a supervised model-visible transfer head repeatedly maps source one-literal nogoods across non-identity variable surface permutations well enough to recover a majority of oracle transfer value while preserving EXP-289 safety floors.

A weaker result blocks EXP-291 under this dependency path and localizes the bottleneck to cross-surface transfer under the frozen representation/objective. It does not erase EXP-289's positive exact episode-local nogood result.
