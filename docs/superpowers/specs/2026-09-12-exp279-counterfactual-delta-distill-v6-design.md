# EXP-279 V6 Counterfactual Branch-Delta Distillation Court — Design

## Status

Pre-data architectural design for DEVELOPMENT-only EXP-279 work. This design starts from clean `main@803fb474eca9cf57713f190e89ef17a113f335e5` after V5 cost-accounted branch preview was falsified and closed without merge.

Frozen protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`, SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

Evidence remains `EV-E2 / UNVERIFIED`. This court may not materialize challenge data, consume confirmatory data, claim promotion, alter MESI/alpha/power/max-n/protected-floor, or consume/reserve provisional fresh DEVELOPMENT evaluation lineage `60000..60032`.

## Scientific question

V3-V5 progressively tested richer selectors and richer pre-branch observability without finding a selector family that had positive direct verified-solution/accounted-FLOP utility at both train60 and train120. V5 additionally showed that paying for 1/2/3 recurrent branch-GRU preview steps is not robustly economic.

The remaining mechanism-level question is whether branch-complementarity information can be **distilled into a cheap representation during augmentation-only training**, so that deployment-time routing does not need to execute branch recurrence merely to decide whether branch recurrence is useful.

V6 tests one primary mechanism only:

`CDD_DELTA_LINEAR`

Counterfactual Delta Distillation (CDD) learns a cheap predictor of the detached full-branch-minus-stop hidden-state delta from features already available before branch execution. A simple linear rescue selector then ranks episodes using the predicted branch delta. The branch computation used to create the teacher exists only on cross-fit training partitions; held-out route features may not use branch recurrence or held-out branch hidden state.

## Why this seam follows V5

V5 failed even when selector features included actual recurrent prefix hidden states, after charging their inference cost. That makes further preview-depth tuning scientifically weak. V6 changes the training signal rather than buying more inference-time recurrent computation:

1. use the canonical branch path as a dense teacher on augmentation training data;
2. distill a hidden counterfactual delta into a cheap feed-forward representation;
3. freeze the distiller before selector training;
4. evaluate direct counterfactual utility with all CDD inference FLOPs charged.

A positive V6 would not prove a deployable router. It would only show that the full branch computation teaches a cheap pre-branch representation with robust economic rescue-ranking value.

## Canonical model boundary

The canonical EXP-279 hybrid is trained unchanged using existing `_train_step` semantics and the existing canonical route threshold `0.5`. After canonical training for a cell, **all canonical parameters are frozen** for the entire V6 court.

V6 may not update:

- event projection;
- variable projection;
- propagation projection;
- reclaimed projection;
- branch GRU;
- decision head;
- verifier head;
- routing head;
- mix gate.

Only fold-local CDD distillers and fold-local selector heads are trainable.

## Data boundary

### Canonical training

Uses only `rng_stream="augmentation"` under the canonical root for the train cell.

### V6 probe data

Independent root:

`20260912-exp279-counterfactual-delta-distill-v6-dev::independent-augmentation-cdd-v6`

Probe data also use only `rng_stream="augmentation"`.

Real court matrix is fixed before data:

- train replicates: `15 / 60 / 120`;
- probe replicates: `198`;
- batch size: `8`;
- cross-fit folds: `3`;
- timesteps: `4`;
- variables: `6`;
- constraints: `3`;
- d_model: `64`;
- hidden size `H`: `48`;
- target parameters: `500000`;
- noise: `0.05`;
- canonical LR: `2e-3`;
- canonical weight decay: `0`;
- distiller steps: `200`;
- distiller LR: `2e-3`;
- distiller weight decay: `0`;
- selector steps: `200`;
- selector LR: `1e-2`;
- selector weight decay: `0`.

Train15 is descriptive only. Train60 and train120 are the only decision cells.

No evaluation RNG stream, evaluation targets, prior consumed DEVELOPMENT lineages, confirmatory examples, or challenge examples may appear in training or feature construction.

## Cross-fit partitioning

Probe replicate `r` is assigned deterministically:

`fold = (r // len(STRATA)) % 3`

For each fold:

- fit partition = the other two folds;
- held-out partition = this fold;
- the distiller is initialized and trained only on fit partition teacher pairs;
- after distiller training, distiller parameters are frozen;
- the selector is initialized and trained only on fit partition rescue labels using distiller predictions;
- held-out scores are produced only after both modules are frozen.

Raw scores are ranked only inside each held-out fold. Raw scores are never compared across independently fitted folds and are never exported.

## Cheap deployment-time inputs

Canonical `_validate_common` produces projected event and variable states. Propagation is computed exactly as in the canonical hybrid.

For an episode define:

- `propagation_state = variables + propagated`;
- `p_mean = mean(propagation_state, variables_dimension)`;
- `e_mean = mean(projected_events, time_dimension)`;
- `e_delta = projected_events[:, -1] - projected_events[:, 0]`.

CDD input is exactly:

`cheap = concat(p_mean, e_mean, e_delta)`

with dimension `3H`.

No branch-GRU output, forced-branch hidden state, target label, rescue label, decision correctness, or verifier output may enter held-out CDD input.

## Dense counterfactual teacher

Teacher generation is permitted only on the fit partition.

For a fit episode:

1. compute canonical stop state exactly as canonical hybrid stop semantics;
2. compute canonical full forced-branch state using the entire branch GRU;
3. detach both states;
4. average their difference over variables:

`teacher_delta = mean(branch_state - stop_state, dim=variables)`

Teacher dimension is `H`.

Teacher branch computation is a **training-only supervision cost** and is not counted as deployment inference cost. V6 makes no training-compute-match claim.

Held-out branch execution may occur only after held-out route scores are frozen, solely to recover the post-hoc stop/forced-branch exact outcomes needed by the non-deployable DEVELOPMENT utility court. Held-out branch hidden state may not be passed to the distiller or selector.

## Distiller architecture

The sole primary distiller family is fixed before data:

```text
Linear(3H, H)
SiLU
Linear(H, H)
```

No layer norm, dropout, residual block, temperature, hidden-width search, extra layer, or alternative activation is allowed.

Loss is full-batch mean squared error on fit-partition detached teacher deltas:

`MSE(predicted_delta, teacher_delta)`

Optimizer is AdamW with LR `2e-3`, weight decay `0`, exactly `200` steps.

The distiller receives no rescue label.

## Rescue selector

After distiller fit, its parameters are frozen.

Selector input is exactly:

`concat(p_mean, predicted_delta)`

with dimension `2H`.

Selector is exactly `Linear(2H, 1)`.

Rescue label remains canonical:

`stop_exact_failure AND forced_branch_exact_success`

Selector uses the same shift-invariant pairwise logistic ranking objective used by prior corrected courts:

`softplus(-(positive_score - negative_score))`

Optimizer is AdamW, LR `1e-2`, weight decay `0`, exactly `200` steps.

Both positive and negative support are required in the selector fit partition. Both positive and negative support are also required in the held-out partition. Any missing support makes the cell `INCONCLUSIVE_SUPPORT`; it may not be coerced to a negative result.

## Descriptive control

A fold-local `RAW_CHEAP_LINEAR` selector over `concat(p_mean, e_mean, e_delta)` may be emitted as a descriptive control using the same pairwise training steps/LR.

It exists only to indicate whether CDD adds ranking value beyond the raw cheap summaries. It **cannot authorize a successor**, cannot satisfy the primary survival rule, and cannot be selected instead of CDD after data visibility.

## CDD inference FLOP accounting

All CDD inference work is charged to every held-out episode.

For hidden size `H`, variables `V`, timesteps `T`:

Cheap summary FLOPs:

- `p_mean`: `V * H`;
- `e_mean`: `T * H`;
- `e_delta`: `H`.

Distiller FLOPs:

- linear `3H -> H`: `2 * 3H * H + H`;
- SiLU: `4 * H`;
- linear `H -> H`: `2 * H * H + H`.

Selector FLOPs:

- linear `2H -> 1`: `4 * H + 1`.

Total CDD inference FLOPs per episode are the sum of all terms above.

For a held-out fold of `N` episodes with oracle route cardinality `k`:

`stop_total = N * C_stop`

`cdd_routed_total = (N-k) * (C_stop + C_CDD) + k * (C_branch + C_CDD)`

Unlike V5 there is no preview reuse term; CDD cost is always additional cheap inference work.

Teacher-generation FLOPs are excluded from deployment utility because teacher computation does not run at inference. The artifact must state this distinction explicitly.

## Direct counterfactual utility court

V6 does not use a posterior break-even threshold.

Within each held-out fold:

1. freeze CDD and selector;
2. score held-out episodes;
3. derive post-hoc stop and forced-branch exact outcomes;
4. define rescue labels;
5. set non-deployable oracle cardinality `k_fold = true rescue count in that held-out fold`;
6. rank only inside the held-out fold;
7. select top `k_fold` scores;
8. selected episodes use forced-branch exact outcome;
9. unselected episodes use stop exact outcome;
10. charge exact path cost plus CDD cost;
11. count selected rescues, selected harms, total exact solutions, and total FLOPs.

Aggregate across folds by summing counts and FLOPs, never by averaging per-fold utility ratios.

Primary direct utility:

`verified_utility = exact_solution_count / total_accounted_flops`

CDD is economically positive only when aggregated CDD-routed utility is **strictly greater** than forced-stop utility.

ROC-AUC and average precision are descriptive only and must be aggregated using fold-local sufficient statistics; they have no survival threshold.

## Cell classifications

Each train cell emits exactly one of:

- `CDD_DIRECT_UTILITY_POSITIVE`
- `CDD_NOT_DIRECTLY_ECONOMIC`
- `INCONCLUSIVE_SUPPORT`

Train15 classification is descriptive only.

## Cross-cell decision

Decision cells are exactly train60 and train120.

Preregistered cross-cell outcomes:

- `ROBUST_CDD_BRANCH_COMPLEMENTARITY`: CDD support is closed and CDD direct utility is strictly positive in **both** train60 and train120;
- `NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY`: support is closed in both cells but one or both cells are not directly economic;
- `INCONCLUSIVE_SUPPORT`: any decision cell lacks required fit-side or held-out support, or fails an evidence-boundary check.

The descriptive raw control can never change this disposition.

## Successor authorization

If cross-cell result is `ROBUST_CDD_BRANCH_COMPLEMENTARITY`:

- authorize only a separately preregistered deployable CDD-router seam;
- do not merge V6 as production routing behavior;
- do not consume/reserve `60000..60032` merely because V6 passed;
- preserve `EV-E2 / UNVERIFIED`.

If cross-cell result is `NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY`:

- close V6 without merge;
- do not continue selector/head/pre-branch-feature tuning under EXP-279;
- the next scientific question must revisit the routing architecture/objective itself or the definition of branch complementarity.

If result is `INCONCLUSIVE_SUPPORT`:

- close without an absence claim;
- diagnose support generation under a new augmentation root before further mechanism claims.

## Required artifact boundary

Every cell artifact and cross-cell receipt must assert:

- `evidence_level = "EV-E2"`;
- `decision = "UNVERIFIED"`;
- `scientific_evidence_eligible = false`;
- `canonical_model_frozen_for_cdd = true`;
- `teacher_uses_fit_partition_only = true`;
- `heldout_branch_hidden_used_for_features = false`;
- `evaluation_rng_stream_used = false`;
- `evaluation_targets_used = false`;
- `confirmatory_data_consumed = false`;
- `challenge_materialized = false`;
- `promotion_claimed = false`;
- `fresh_evaluation_lineage_may_be_reserved = false`;
- `fresh_evaluation_lineage_consumed = false`;
- `cdd_inference_flops_charged_to_primary_utility = true`;
- `teacher_training_flops_excluded_from_deployment_utility = true`;
- `raw_scores_compared_across_folds = false`;
- `raw_scores_exported = false`.

## Engineering process

Before any real V6 matrix is visible:

1. core CDD/teacher/cost/direct-utility tests must establish RED, then GREEN;
2. CLI tests must establish a separate RED, then GREEN;
3. cross-cell anti-cherry-pick classifier tests must establish a separate RED, then GREEN;
4. dedicated V6 workflow contract must be GREEN before scientific jobs may run;
5. workflow must freeze all real matrix values and boundary assertions before the first real V6 artifact is emitted.

After scientific data become visible, scientific code, budgets, root, loss, architecture, and disposition rule may not be changed on this lineage.
