# EXP-279 V5 — Cost-Accounted Branch-Preview Court

Date: 2026-09-11
Status: design approved by standing user authorization; frozen before implementation/data
Base: `main@803fb474eca9cf57713f190e89ef17a113f335e5`
Evidence boundary: DEVELOPMENT only (`EV-E2 / UNVERIFIED`)

## 1. Scientific question

EXP-279 has now falsified robust rescue selection from the frozen propagation representation and from cheap pre-branch summaries of projected surface events. The branch path nevertheless has sparse post-hoc rescue capacity and obtains information unavailable to the route decision by executing `branch_gru` over the projected event sequence.

V5 asks one narrower mechanism question:

> Does a **partial, reusable execution of the canonical branch GRU** reveal enough branch-complementarity information to improve exact-solution / accounted-FLOP utility once the preview and selector costs are charged?

This is not a deployable routing claim. It is an upper-bound observability/economics court that determines whether a cost-accounted branch-preview successor is worth designing at all.

## 2. Non-goals and immutable boundaries

V5 will not:

- modify canonical EXP-279 training;
- modify the canonical route threshold (`0.5`);
- modify MESI, alpha, power, max-n, protected floor, or frozen protocol bytes;
- consume evaluation or confirmatory examples/targets;
- consume or reserve `60000..60032`;
- tune preview depth after observing V5 data;
- search selector architectures, margins, class weights, temperatures, priors, replay ratios, or thresholds;
- compare raw scores across independently fitted folds;
- claim the oracle route cardinality is deployable;
- claim promotion, challenge materialization, or confirmatory evidence.

## 3. Frozen canonical model

Train the canonical hybrid exactly as current `main`, then freeze all canonical parameters before V5 probe fitting.

For each independent augmentation-only probe batch:

- `events = SiLU(event_projection(surface_events))`;
- `variables = SiLU(variable_projection(variable_states))`;
- `propagated = _propagate(variables, incidence)`;
- `propagation_state = variables + propagated`;
- `p_mean = mean(propagation_state, dim=variables)`.

Forced-stop and forced-full-branch exact outcomes are obtained with the existing same-weight helpers. The rescue label remains exactly:

`rescue = stop_exact_failure AND forced_branch_exact_success`.

## 4. Preview mechanism and resume equivalence

The only preview computation is the **existing canonical** `branch_gru` applied to an event prefix.

For preview depth `p`:

`_, h_p = branch_gru(events[:, :p, :])`.

If an episode is later routed, the remaining suffix is run using `h_p` as the initial hidden state:

`_, h_T = branch_gru(events[:, p:, :], h_p)`.

The implementation contract must verify that prefix+suffix resume produces the same final hidden state as one full `branch_gru(events)` execution within deterministic floating-point tolerance. This justifies treating preview FLOPs as **reusable** on routed episodes instead of double-counting them.

No alternative GRU, learned preview encoder, event reordering, mean-event shortcut, or hidden-state transformation is allowed in V5.

## 5. Frozen preview families

Exactly three families are allowed, differing only in prefix depth:

- `PREFIX1_STATE_LINEAR`: first 1 of 4 event steps;
- `PREFIX2_STATE_LINEAR`: first 2 of 4 event steps;
- `PREFIX3_STATE_LINEAR`: first 3 of 4 event steps.

For every family, selector input is:

`concat(p_mean, preview_hidden)` with dimension `2H`.

The selector is exactly one linear layer:

`Linear(2H, 1)`.

No nonlinearity, second hidden layer, calibration layer, or architecture search is allowed. This keeps the comparison focused on **preview information depth**, not head capacity.

## 6. Probe supervision and cross-fitting

Use 3-fold cross-fitting on a new independent augmentation root. Each selector is independently initialized and fitted on two folds, then scored only on the held-out fold.

Pairwise ranking loss is fixed:

`softplus(-(positive_score - negative_score))`.

Positive = exact rescue. Negative = every non-rescue episode.

Both the fit partition and held-out partition of every fold must contain at least one positive and one negative. Otherwise support is open and the cell is inconclusive.

Because pairwise loss is shift-invariant, raw selector scores from different folds may never be globally ranked or calibrated. All routing decisions remain fold-local.

## 7. Non-deployable oracle route cardinality

Within each held-out fold:

`k_fold = true rescue count in that held-out fold`.

The family ranks only episodes in that fold and routes the top `k_fold`.

This cardinality uses held-out labels and is explicitly non-deployable. It exists only to ask whether the preview representation contains an economically useful high-ranking tail under an optimistic selector-cardinality oracle.

Artifacts must state:

- `oracle_route_cardinality_is_deployable=false`;
- `heldout_targets_used_for_route_cardinality=true`;
- `successor_must_replace_oracle_cardinality=true`.

## 8. Exact cost accounting

Let:

- `C_stop` = sealed hybrid stop accounted FLOPs per episode;
- `C_branch` = sealed hybrid full-branch accounted FLOPs per episode;
- `G(p)` = canonical GRU FLOPs for `p` recurrent steps using the existing EXP-279 analytical GRU formula;
- `C_pool = variables * hidden_size` for `p_mean` reduction under the repository scalar-arithmetic convention;
- `C_head = 2*(2*hidden_size)*1 + 1` for `Linear(2H,1)`;
- `C_selector = C_pool + C_head`.

For a held-out fold with `N` episodes and `k` selected episodes:

- every episode pays `C_selector`;
- every unselected episode pays `C_stop + G(p) + C_selector`;
- every selected episode pays `C_branch + C_selector`.

Selected episodes do **not** pay `G(p)` twice because the prefix hidden is reused and resume equivalence is a tested invariant.

Therefore:

`total_preview_routed_flops = (N-k)*(C_stop + G(p) + C_selector) + k*(C_branch + C_selector)`.

Forced-stop baseline remains:

`total_stop_flops = N * C_stop`.

This deliberately makes V5 harder than V4: preview/selector overhead is charged against the primary utility.

## 9. Direct counterfactual utility

For selected held-out episodes use the exact forced-full-branch outcome; for unselected episodes use exact forced-stop outcome.

Count directly:

- stop exact solutions;
- routed counterfactual exact solutions;
- selected rescues;
- selected harms (`stop succeeds AND branch fails`);
- selected neutral routes;
- total stop FLOPs;
- total preview-routed FLOPs.

Across folds, aggregate counts and total FLOPs, never raw scores and never averages of fold utility ratios.

Primary metric:

`U_stop = total_stop_solution_count / total_stop_flops`

`U_preview = total_counterfactual_solution_count / total_preview_routed_flops`

A preview family passes one train cell iff support is closed and:

`U_preview > U_stop` strictly.

AUC, average precision, selector precision, losses, and route fraction are descriptive only.

## 10. Frozen DEVELOPMENT budget

Root:

`20260911-exp279-cost-accounted-branch-preview-v5-dev`

Independent probe suffix:

`::independent-augmentation-preview-v5`

Frozen matrix:

- canonical train replicates: `15 / 60 / 120`;
- train=15: descriptive only;
- decision cells: train=60 and train=120;
- probe replicates per cell: `198`;
- batch size: `8` (`1,584` probe episodes/cell);
- folds: `3`;
- pairwise selector steps/fold: `200`;
- selector learning rate: `1e-2`;
- selector weight decay: `0`;
- `d_model=64`;
- `hidden_size=48`;
- matched target parameters: `500000`;
- timesteps: `4`;
- variables: `6`;
- constraints: `3`;
- noise std: `0.05`;
- canonical learning rate: `2e-3`;
- canonical weight decay: `0`;
- canonical route threshold: `0.5`.

No budget or family may change after real V5 matrix data are observed.

## 11. Preregistered cross-cell decision

The same family must have positive direct utility at **both train=60 and train=120**.

Decision hierarchy prefers the cheapest surviving preview:

1. `ROBUST_PREFIX1_BRANCH_PREVIEW` if `PREFIX1_STATE_LINEAR` passes both decision cells;
2. else `ROBUST_PREFIX2_BRANCH_PREVIEW` if `PREFIX2_STATE_LINEAR` passes both;
3. else `ROBUST_PREFIX3_BRANCH_PREVIEW` if `PREFIX3_STATE_LINEAR` passes both;
4. else `NO_ROBUST_COST_ACCOUNTED_BRANCH_PREVIEW` if support is closed;
5. `INCONCLUSIVE_SUPPORT` if any required family/fold lacks fit-side or held-out support.

Mixed-family success across train60/train120 is negative, not robust.

`successor_design_authorized=true` only for the three `ROBUST_*` decisions.

Even for a robust positive result:

- `fresh_evaluation_lineage_may_be_reserved=false`;
- `fresh_evaluation_lineage_consumed=false`;
- `60000..60032` remains untouched.

A positive result only authorizes a separately designed deployable preview router with non-oracle routing policy and exact selector/preview cost accounting.

## 12. Negative-result successor policy

If no prefix depth survives both decision cells, V5 is falsified and closes without merge. Do not add prefix-4 after observing V5: full four-step recurrence for every episode defeats the purpose of routing and is not a legitimate post-hoc family extension.

The next legitimate seam after a negative V5 is targeted **representation learning for branch-complementarity observability** under a new independent augmentation root, not further preview-depth or selector-head feature fishing.

## 13. Artifact contract

Every V5 cell artifact must include:

- schema/version;
- `EV-E2 / UNVERIFIED`;
- frozen protocol id/digest;
- source/code digest;
- canonical root and independent probe-root provenance;
- augmentation-only RNG declarations;
- frozen canonical functional-state digest;
- preview family/depth semantics;
- resume-equivalence assertion metadata;
- `branch_gru_used_for_preview=true`;
- `preview_hidden_reused_for_selected_branch=true`;
- fit/held-out support counts for each fold;
- fold sufficient statistics and aggregate direct utility;
- exact `C_stop`, `C_branch`, `G(p)`, `C_pool`, `C_head`, `C_selector` receipts;
- `probe_inference_flops_charged_to_primary_utility=true`;
- `preview_flops_charged_to_primary_utility=true`;
- `raw_scores_exported=false`;
- `raw_scores_compared_across_folds=false`;
- `evaluation_rng_stream_used=false`;
- `evaluation_targets_used=false`;
- `fresh_evaluation_lineage_may_be_reserved=false`;
- `fresh_evaluation_lineage_consumed=false`;
- `confirmatory_data_consumed=false`;
- `challenge_materialized=false`;
- `promotion_claimed=false`;
- `scientific_evidence_eligible=false`;
- canonical artifact digest.

Validators fail closed on any drift.

## 14. TDD / release order

Separate RED -> GREEN gates are required for:

1. preview/resume + exact cost/direct-utility core;
2. CLI/frozen-protocol/refuse-overwrite behavior;
3. cross-cell anti-cherry-pick classifier;
4. real matrix workflow release.

No V5 matrix cell may run before all pre-data contracts are GREEN. Scientific jobs must depend on the contract gate.

Final interpretation requires fresh exact-head success for:

- dedicated V5 CI;
- EXP-279 confirmatory contract;
- generic repository CI including model-smoke.

## 15. Scientific invariants

Throughout V5:

- protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`;
- protocol SHA256 remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`;
- evidence remains `EV-E2 / UNVERIFIED`;
- no confirmatory data are consumed;
- no challenge is materialized;
- no promotion is claimed;
- `60000..60032` remains untouched;
- V5 may authorize successor design only.