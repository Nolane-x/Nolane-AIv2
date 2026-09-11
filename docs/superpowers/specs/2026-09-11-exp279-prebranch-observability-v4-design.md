# EXP-279 V4 — Cheap Pre-Branch Observability Court

Date: 2026-09-11
Status: written design ready for user review before implementation
Base: `main@803fb474eca9cf57713f190e89ef17a113f335e5`
Evidence boundary: DEVELOPMENT only (`EV-E2 / UNVERIFIED`)

## 1. Problem statement

EXP-279 has established two facts that must be preserved simultaneously:

1. the same-weight branch path has sparse, real, post-hoc rescue value over the stop path;
2. deployable routing attempts based on the current propagation-state routing representation have not produced robust positive utility.

The most recent valid court, V3, repaired earlier methodological defects by ranking only inside held-out folds and evaluating selector quality with direct counterfactual exact-solution / path-dependent-FLOP utility. V3 found an isolated positive `DEEPSETS` result at train=60 but no robust same-family signal at train=120, so it closed without merge.

The code path reveals a concrete observability asymmetry. The current route decision is computed from `propagation_state = variables + propagated`. The branch path, however, obtains additional information from the projected surface-event sequence through `branch_gru`. The route decision is made before that recurrent branch computation. Therefore the next question is not whether another loss, class weight, prior, threshold, or richer head over the same propagation representation can work. The next question is whether **cheap information already available before branch execution** contains enough branch-complementarity signal to identify rescue episodes.

V4 is a diagnostic court for that question only.

## 2. Goals

V4 must determine whether sparse branch rescues are economically selectable from frozen, pre-branch observables without executing the branch GRU.

A positive V4 result may authorize design of a deployable router that consumes cheap surface-event / propagation features. It does **not** authorize confirmatory execution, challenge materialization, promotion, or consumption of the provisional fresh DEVELOPMENT evaluation lineage `60000..60032`.

A negative V4 result means the current cheap pre-branch observables do not provide a robust enough selector signal under this court. The successor should then investigate representation learning or an explicitly cost-accounted branch-preview mechanism rather than continuing to tune routing losses or thresholds.

## 3. Non-goals

V4 will not:

- modify canonical EXP-279 training;
- modify the canonical route threshold (`0.5`);
- modify MESI, alpha, power, max-n, protected floor, or frozen protocol bytes;
- execute `branch_gru` to create selector features;
- tune class weights, replay ratios, temperatures, posterior priors, economic multipliers, or routing thresholds;
- use evaluation or confirmatory examples/targets;
- compare raw scores across independently trained folds;
- claim that a diagnostic probe itself is a deployable routing head;
- open or consume `60000..60032` even if V4 passes.

## 4. Alternatives considered

### 4.1 Richer head on propagation state only

Rejected as the primary successor. V3 already tested linear, MLP-mean, and DeepSets probes over frozen propagation states and failed the preregistered cross-cell robustness rule.

### 4.2 Partial branch-GRU preview

Deferred. A recurrent preview may expose useful branch-specific signal, but it spends part of the very branch computation whose economic value routing is meant to protect. It introduces a new cost-accounting question and is inappropriate before testing observables already available before branch execution.

### 4.3 Cheap pre-branch observability court — selected

Use projected surface events that already exist before branch execution, combine them with the frozen propagation representation in a small fixed hierarchy, and evaluate selector quality with V3's direct counterfactual utility semantics.

## 5. Frozen canonical model and feature boundary

Train the canonical hybrid exactly as current `main`, then freeze all canonical parameters before fitting any probe.

For a frozen canonical hybrid and an augmentation batch:

- `events = SiLU(event_projection(surface_events))`, shape `[B, T, H]`;
- `variables = SiLU(variable_projection(variable_states))`;
- `propagated = _propagate(variables, incidence)`;
- `propagation_state = variables + propagated`, shape `[B, V, H]`.

The V4 feature extractor must **not call** `_branch_context`, `branch_gru`, or `HybridRoutingArm.forward` in a way that conditionally executes branch recurrence for selector feature construction. The existing forced-branch helper may execute the branch GRU only to produce the explicitly non-deployable training/diagnostic rescue label and counterfactual outcome; those outcomes are never fed back as selector features.

Pre-branch summaries are fixed before data:

- `p_mean = mean(propagation_state, dim=variables)`;
- `e_mean = mean(events, dim=time)`;
- `e_delta = events[:, -1, :] - events[:, 0, :]`.

`event_projection` is already part of the canonical shared model path and is therefore an allowed pre-branch observable. `e_delta` is the single preregistered temporal summary; no additional temporal statistics may be added after observing V4 results.

## 6. Probe families

The probe family is fixed to exactly three members.

### 6.1 `STATE_DEEPSETS`

Continuity baseline matching the strongest propagation-state-only family from V3.

- input: full `propagation_state [B,V,H]`;
- architecture: `phi: Linear(H,H) -> SiLU`, mean across variables, then `rho: Linear(H,H) -> SiLU -> Linear(H,1)`.

### 6.2 `STATE_EVENT_MEAN`

Tests whether adding a cheap order-invariant surface-event summary exposes rescue signal.

- input vector: `concat(p_mean, e_mean)` (`2H`);
- architecture: `Linear(2H,H) -> SiLU -> Linear(H,1)`.

### 6.3 `STATE_EVENT_RESIDUAL`

Tests whether explicit cross-modal mismatch plus one cheap temporal direction exposes branch-complementarity evidence.

- input vector: `concat(p_mean, e_mean, e_delta, p_mean - e_mean, p_mean * e_mean)` (`5H`);
- architecture: `Linear(5H,H) -> SiLU -> Linear(H,1)`.

No family may be added, removed, renamed, or altered after V4 data are observed. No automated architecture search or feature search is allowed.

## 7. Probe supervision and cross-fitting

The rescue label remains exact and unchanged:

`rescue = stop_exact_failure AND forced_branch_exact_success`.

Forced-stop and forced-branch exact outcomes are obtained from the frozen canonical hybrid using the existing same-weight helpers. Evaluation targets are never used; all V4 examples come from an independent augmentation stream.

Use 3-fold cross-fitting. Each probe is independently initialized and fitted on two folds and scored only on the held-out fold. Pairwise ranking loss remains:

`softplus(-(positive_score - negative_score))`.

For every fold and every probe family, **both** the probe-fit partition and the held-out partition must contain at least one rescue and at least one non-rescue. If any required partition lacks support, that train cell is support-inconclusive and cannot be classified as positive or negative.

Because the pairwise loss is shift-invariant, raw logits from different folds must never be globally ranked or calibrated. Every selector decision is fold-local.

## 8. Route cardinality and direct counterfactual utility

V4 retains the explicitly non-deployable oracle diagnostic cardinality from V3:

`k_fold = true rescue count in that held-out fold`.

Within each held-out fold and probe family:

1. rank probe scores only inside that fold;
2. select the top `k_fold` episodes;
3. selected episodes use the exact forced-branch outcome;
4. unselected episodes use the exact forced-stop outcome;
5. count selected rescues, selected harms, selected neutral routes, stop exact solutions, and routed counterfactual exact solutions;
6. charge `C_stop` for unselected episodes and `C_branch` for selected episodes from the sealed matched-arm ledger;
7. emit fold sufficient statistics and fold-local descriptive ranking diagnostics.

Across folds, aggregate **counts and FLOPs**, never raw scores and never averages of per-fold utility ratios:

`U_stop = total_stop_solution_count / total_stop_accounted_flops`

`U_routed = total_counterfactual_solution_count / total_counterfactual_accounted_flops`

A probe is positive in one train cell iff:

- all probe-fit and held-out partitions satisfy the support requirement above; and
- `U_routed > U_stop` strictly.

AUC, average precision, and selected-rescue precision are descriptive only. There is no AUC floor, AP floor, posterior break-even threshold, or calibrated probability threshold in the decision rule.

## 9. Probe-compute interpretation

V4 is an **observability upper-bound court**, not a deployable-router benchmark. The primary direct counterfactual utility charges canonical stop-vs-branch path FLOPs exactly as V3 and does not charge the diagnostic probe's own inference FLOPs.

To prevent this from being misread as a deployable economics claim, every artifact must state:

- `probe_is_deployable=false`;
- `probe_inference_flops_charged_to_primary_utility=false`;
- `successor_must_account_selector_flops=true`.

Probe parameter counts and analytical probe FLOPs should be emitted descriptively. A positive V4 result authorizes a successor design only; that successor must implement and charge the actual selector before any fresh evaluation lineage can be considered.

## 10. Frozen DEVELOPMENT budget

Use a new independent augmentation root:

`20260911-exp279-prebranch-observability-v4-dev`

Probe root suffix:

`::independent-augmentation-probe-v4`

Frozen matrix:

- canonical train replicates: `15 / 60 / 120`;
- train=15: descriptive only;
- decision cells: train=60 and train=120;
- probe replicates per cell: `198`;
- batch size: `8` (`1,584` probe episodes/cell);
- folds: `3`;
- pairwise probe steps per fold: `200`;
- probe learning rate: `1e-2`;
- probe weight decay: `0`;
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

No V4 budget parameter may be changed after real V4 matrix data are observed.

## 11. Preregistered cross-cell decision

The same family must survive both decision cells. Train=15 cannot authorize anything.

Allowed cross-cell decisions:

- `ROBUST_STATE_ONLY_PREBRANCH_SIGNAL`: `STATE_DEEPSETS` has positive direct utility at both train=60 and train=120.
- `ROBUST_EVENT_MEAN_PREBRANCH_SIGNAL`: state-only is not robust, but `STATE_EVENT_MEAN` has positive direct utility at both train=60 and train=120.
- `ROBUST_EVENT_RESIDUAL_PREBRANCH_SIGNAL`: neither simpler family is robust, but `STATE_EVENT_RESIDUAL` has positive direct utility at both train=60 and train=120.
- `NO_ROBUST_PREBRANCH_OBSERVABILITY`: all required support is closed but no single family is direct-utility-positive in both decision cells.
- `INCONCLUSIVE_SUPPORT`: either decision cell lacks required fit/held-out positive/negative support for any required fold/probe family.

The ordering above prevents cherry-picking a more complex family when a simpler family already survives.

`successor_design_authorized=true` only for the three `ROBUST_*` outcomes. Even then:

- `fresh_evaluation_lineage_may_be_reserved=false`;
- `fresh_evaluation_lineage_consumed=false`;
- `60000..60032` remains untouched.

## 12. Artifact boundary and validation

Every V4 cell artifact must include:

- schema/version;
- `EV-E2 / UNVERIFIED`;
- frozen protocol id and digest;
- canonical code/source digest;
- root/probe-root provenance;
- augmentation-only RNG declarations;
- frozen canonical final-state digest;
- feature semantics and explicit `branch_gru_used_for_probe_features=false`;
- per-probe/per-fold fit-support and held-out-support receipts;
- per-probe/per-fold sufficient statistics;
- aggregate direct utility;
- probe parameter/FLOP diagnostics;
- `raw_scores_exported=false`;
- `raw_scores_compared_across_folds=false`;
- `evaluation_rng_stream_used=false`;
- `evaluation_targets_used=false`;
- `fresh_evaluation_lineage_consumed=false`;
- `confirmatory_data_consumed=false`;
- `challenge_materialized=false`;
- `promotion_claimed=false`;
- `scientific_evidence_eligible=false`;
- canonical artifact digest.

Validators must fail closed if any boundary drifts.

## 13. TDD and execution plan requirements

Implementation must use separate RED -> GREEN cycles for:

1. feature-construction and direct-utility court core;
2. CLI runner / frozen-protocol / refuse-overwrite behavior;
3. cross-cell anti-cherry-pick classifier;
4. workflow release of the real augmentation-only matrix.

The real `15/60/120` matrix must not run until all pre-data contracts are GREEN. Workflow dependency must make scientific jobs `needs: contract` so a contract failure skips data release.

Fresh generic repository CI and focused EXP-279 confirmatory-contract CI must be green on the exact final scientific head before merge or closure is interpreted as an engineering-clean scientific result.

## 14. Interpretation and successor policy

### If V4 is robust positive

Do not merge a deployable router immediately and do not consume `60000..60032`. The next seam is a separately designed, TDD-built deployable selector using only the winning cheap feature family, with explicit selector-FLOP accounting and a new preregistered DEVELOPMENT evaluation lineage.

### If V4 is negative

Do not add more hand-selected cheap features on the same observed root. Close V4 without merge as negative DEVELOPMENT evidence. The next legitimate seam is representation learning for branch-complementarity observability or an explicitly cost-accounted branch-preview mechanism under a new design and independent augmentation root.

### If V4 is inconclusive

Do not infer absence of signal. Diagnose support-generation geometry separately without changing V4's decision rule post hoc.

## 15. Scientific invariants

Throughout V4:

- protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`;
- protocol SHA256 remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`;
- evidence remains `EV-E2 / UNVERIFIED`;
- no confirmatory data are consumed;
- no challenge is materialized;
- no promotion is claimed;
- `60000..60032` remains untouched;
- V4 results may authorize successor design only.