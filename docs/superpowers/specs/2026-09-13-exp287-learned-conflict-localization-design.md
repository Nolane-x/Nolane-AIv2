# EXP-287 Learned Conflict Localization Court — Design

## 1. Scientific question

EXP-286 established scoped EV-E3 evidence that generator-exact conflict-core information can reduce reasoning cost under the frozen synthetic conflict family. EXP-287 asks the next narrower question from V0.16.1:

> Can a learned conflict localizer, using only ordinary model-visible state at the current contradiction, recover a material fraction of the same-weights oracle conflict-core value without violating the protected solution-rate floor?

This is a DEVELOPMENT court first. It does not claim safe dependency-directed backjumping, learned clause transfer, general natural-language conflict localization, or integrated NLM success. EXP-288 remains blocked unless this court closes recurrently positive.

Base: `main@803fb474eca9cf57713f190e89ef17a113f335e5`.

Frozen Stage-A protocol lineage digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`. The Stage-A protocol bytes are not modified by EXP-287.

## 2. Why a same-weights three-mode court

The decisive uncertainty is localization, not whether oracle cores have value. Training three independent end-to-end agents would mix localization quality with optimizer variance, representation drift, and control-policy adaptation. EXP-287 therefore trains one canonical conflict model per root and evaluates the frozen identical weights in three information modes:

1. `NULL_CORE_CONTROL` — localizer is computed and charged, but its output is ignored and the conflict token is all zero.
2. `LEARNED_CORE_TOP2` — localizer is computed from model-visible state; its two highest-scoring variables become the conflict token after the current contradiction.
3. `ORACLE_CORE_UPPER_BOUND` — the same localizer computation is still executed and charged, but the generator-exact two-variable core replaces its output after the current contradiction.

The only evaluation-time difference is which post-contradiction conflict token enters the already-trained shared rollback/verifier state. This makes the learned-versus-oracle gap interpretable as localization loss under the frozen synthetic family.

`ORACLE_CORE_UPPER_BOUND` is explicitly non-deployable. Evaluation core labels may enter only this oracle mode and post-hoc localization metrics. They may not train, calibrate, threshold, select, or otherwise alter `LEARNED_CORE_TOP2`.

## 3. Model geometry

EXP-287 reuses the tested EXP-286 conflict-world geometry but adds a learned localizer head inside an EXP-287-specific model. Per canonical root:

- `d_model = 64`
- `hidden_size = 48`
- `target_parameters = 500000`
- `timesteps = 4`
- `variables = 8`
- `decoys = 3`
- true core size = exactly `2` by the frozen synthetic generator
- `max_search_steps = 16`
- `batch_size = 8`
- `noise_std = 0.05`
- optimizer = AdamW
- `lr = 0.002`
- `weight_decay = 0.0`

The localizer consumes only the pre-core shared state built from projected surface events, projected variable states, and recurrent event context. For variable `i`:

`localizer_logit_i = Linear(hidden_state_i)`

No solution target, generator metadata, core mask, decoy order truth, evaluator truth, future contradiction, or oracle receipt is an input to this head.

All three evaluation modes execute the localizer head on every neural search step, so localizer FLOPs are charged even when ignored by control/oracle modes. Parameter, active-functional-parameter, optimizer-visible-parameter, and analytical neural-compute ledgers must match exactly across the three modes.

## 4. Training contract

Each canonical root trains exactly one model. Training uses only fresh EXP-287 `augmentation` worlds. Evaluation worlds are disjoint and unopened until the model is frozen.

At each augmentation example after the current contradiction, training uses:

- rollback target: generator-defined conflict variable already used by the EXP-286 synthetic court;
- verifier target: augmentation solution target;
- localizer target: augmentation `core_mask` membership;
- core conditioning for the shared rollback/verifier path: the augmentation ground-truth core mask.

Total loss is fixed before data visibility:

`loss = rollback_cross_entropy + verifier_binary_cross_entropy + localizer_binary_cross_entropy`

All three terms have coefficient exactly `1.0`. There are no class weights, focal terms, temperature, threshold sweeps, calibration steps, curriculum switches, replay, hard-negative mining, early stopping, or post-result loss-weight changes.

Using augmentation core labels is allowed because EXP-287 studies whether conflict localization can be learned from supervised synthetic conflict examples. Evaluation core labels remain hidden from the learned mode.

## 5. Frozen DEVELOPMENT geometry

The court uses four independent canonical roots to reject one-seed localization wins:

- canonical indices: `0,1,2,3`
- root prefix: `20260913-exp287-learned-conflict-localization-v1-dev`
- `train_replicates = 64` per canonical root
- `eval_replicates = 32` per canonical root
- each replicate contains batch size `8`
- training episodes/root: `512`
- evaluation episodes/root: `256`
- total across four roots: `2,048` training episodes and `1,024` held-out evaluation episodes

Per canonical root, model initialization, augmentation, and evaluation seeds are independently derived through the repository seed-domain machinery. Training and evaluation replicate namespaces are disjoint. No EXP-286 confirmatory observation and no EXP-279 V1–V10 observation is reused for fitting or selection.

Evaluation replicate data is read only after the root model state and all fixed decision constants are frozen.

## 6. Learned conflict token

`LEARNED_CORE_TOP2` uses exactly the two largest localizer logits after the current contradiction. Ties are broken by ascending variable index. The selected variables form a binary two-hot mask.

There is no probability threshold and no variable-cardinality sweep. The top-2 rule is scoped to this synthetic family because its generator defines exactly two-variable minimal cores. Passing this court would not authorize assuming core cardinality two outside this family.

Before contradiction, every mode receives the same null conflict token. Pre-contradiction conflict delivery is a hard validation failure.

## 7. Search semantics

Search reuses the EXP-286 chronological/oracle discipline with one generalization: at the contradiction, `NULL_CORE_CONTROL` performs chronological rollback behavior; learned/oracle modes order their selected two variables by the shared rollback logits and place them at the front of the remaining queue.

The external evaluator remains responsible for solution verification. Censored episodes are retained and charged at the declared maximum accounted-FLOP ceiling. No evaluator truth may choose a learned action.

A per-step receipt records mode, selected conflict variables, whether oracle information was delivered, visited variable, contradiction status, accounted FLOPs, and external verification outcome. `LEARNED_CORE_TOP2` receipts must assert `oracle_information_delivered=false` for every episode.

## 8. Primary economic metric

The primary DEVELOPMENT metric keeps EXP-286 semantics: accounted reasoning FLOPs to verified solution, lower is better. For each canonical root let:

- `C0` = mean accounted cost of `NULL_CORE_CONTROL`
- `CL` = mean accounted cost of `LEARNED_CORE_TOP2`
- `CO` = mean accounted cost of `ORACLE_CORE_UPPER_BOUND`

Define oracle headroom:

`H_oracle = C0 - CO`

and learned recovered headroom:

`H_learned = C0 - CL`.

When `H_oracle > 0`, define oracle-value capture:

`capture = H_learned / H_oracle`.

The value is not clipped. Negative learned benefit therefore produces negative capture; learned performance better than oracle can exceed one and is retained rather than normalized away.

## 9. Localization diagnostics

Held-out core labels may be used only after actions are fixed to compute diagnostics:

- top-2 core precision = selected true-core members / 2;
- top-2 core recall = selected true-core members / 2 (equal to precision under fixed cardinality, retained explicitly for audit);
- exact-core recovery rate;
- off-core selection rate;
- variable-wise average precision from localizer logits;
- raw core prevalence (`2/8 = 0.25`).

These metrics diagnose K215 noise. They never tune the top-2 rule or loss.

## 10. Root classification

A canonical root is `LEARNED_LOCALIZATION_VALUE_ESTABLISHED` only if all predicates hold:

1. `H_oracle > 0` — the fresh root reproduces positive oracle headroom;
2. `H_learned > 0` — learned localization reduces accounted cost versus same-weights null control;
3. `capture >= 0.50` — learned localization recovers at least half of fresh same-weights oracle cost headroom;
4. learned verified-solution rate is at least control verified-solution rate minus `0.005`;
5. learned top-2 core precision is strictly greater than `0.50` (more than one true core member selected on average);
6. learned evaluation receives no oracle information and no pre-contradiction conflict token.

If predicate 1 fails, classification is `ORACLE_HEADROOM_NOT_REPLICATED`; this is not converted into evidence against the learned localizer because the denominator needed for “approach oracle value” is absent.

If predicate 1 passes but any predicate 2–6 fails, classification is `LEARNED_LOCALIZATION_VALUE_NOT_ESTABLISHED`.

The `0.50` capture threshold and `0.50` top-2 precision threshold are fixed design commitments, not learned or post-hoc estimates. They encode the minimum requirement that a successor recover a majority of the oracle signal in both economic and localization terms before any backjump design is justified.

## 11. Cross-root decision

The four roots are all decision roots; there is no post-hoc root selection.

- `LOCALIZATION_VALUE_RECURRENT` only if all four roots are `LEARNED_LOCALIZATION_VALUE_ESTABLISHED`.
- `LOCALIZATION_VALUE_INTERMITTENT` if at least one but not all four roots establishes value.
- `LOCALIZATION_VALUE_NOT_ESTABLISHED` if zero roots establish value and all four reproduced positive oracle headroom.
- `ORACLE_REPLICATION_INCOMPLETE` if any root is `ORACLE_HEADROOM_NOT_REPLICATED`.

Only `LOCALIZATION_VALUE_RECURRENT` may set `successor_design_authorized=true`, with exact scope `DESIGN_EXP288_BACKJUMP_COURT_ONLY`.

Every other decision sets `successor_design_authorized=false`.

No EXP-287 outcome directly authorizes an EXP-288 implementation, confirmatory data, challenge materialization, Stage-A protocol modification, semantic authority, or promotion. A positive EXP-287 court authorizes only a separately preregistered EXP-288 design.

## 12. Evidence boundary

EXP-287 V1 is `EV-E2 / DEVELOPMENT` regardless of outcome.

Every receipt must record:

- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_labels_used_for_training = false`
- `evaluation_core_used_by_learned_mode = false`
- `oracle_mode_deployable = false`

The court may report a DEVELOPMENT mechanism result; it may not report a scientific NLM capability claim.

## 13. Integrity and serialization

All root and cross receipts use canonical JSON with string keys from construction time. Each artifact contains `receipt.json`, a matching SHA-256 sidecar, code/protocol identity, root identity, and sufficient aggregate/episode receipts needed to recompute classification.

Validators recompute every decision predicate from primitive fields and reject forged classification labels, mismatched geometry, root duplication, model-state drift, evaluation/training lineage overlap, oracle leakage, boundary overclaim, and canonical serialize→parse→serialize drift.

The learned model digest is captured after training and must remain byte-identical across all three evaluation modes.

## 14. No post-result tuning

After any EXP-287 held-out evaluation result becomes visible, V1 is frozen. The following are prohibited inside V1:

- changing top-k cardinality;
- changing capture or precision thresholds;
- changing loss weights;
- adding class weighting/focal loss/calibration;
- increasing sample counts;
- dropping a canonical root;
- changing localizer geometry;
- rerunning a failed root under a new random seed as replacement;
- consuming confirmatory/challenge data.

An instrumentation defect that prevents valid receipt generation may be repaired only through a new explicitly versioned execution attempt that preserves scientific semantics and documents the invalid attempt. An unfavorable valid result is not an instrumentation defect.

## 15. Success and failure interpretation

`LOCALIZATION_VALUE_RECURRENT` means only that, on this frozen synthetic conflict family, a supervised learned localizer repeatedly recovers a majority of same-weights oracle conflict-core cost value while preserving the protected solution-rate floor. It justifies designing EXP-288.

Any weaker result blocks EXP-288 under the current hypothesis. It does not erase EXP-286 oracle value; it localizes the bottleneck to learned conflict localization under this representation/objective.
