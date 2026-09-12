# EXP-279 V10 Counterfactual Representation Identifiability Court — Design

Date: 2026-09-12
Status: PRE-DATA / DEVELOPMENT DIAGNOSTIC / NOT SCIENTIFICALLY RELEASED
Base: `main@803fb474eca9cf57713f190e89ef17a113f335e5`
Protocol: `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`
Protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
Evidence: `EV-E2 / UNVERIFIED`
Court name: **CRIC — Counterfactual Representation Identifiability Court**

## 1. Research question

EXP-279 V9 failed its frozen quartet mechanism court: train60 and train120 were both `QUARTET_POLICY_NOT_ECONOMIC`, all eight canonical roots were `QUARTET_ROOT_NOT_ECONOMIC`, and no successor authorization was opened. The sealed V9 outcome also showed a distinct diagnostic fact: a perfect rescue-only oracle would have positive utility gain over STOP, while the deployed V9 cheap policy routed zero episodes across all roots.

V10 does **not** repair, tune, rerun, or reuse V9. It asks a different upstream question:

> Does the frozen EXP-279 hybrid contain enough target-relevant information in cheap pre-branch or stop-frontier internal state to identify branch-rescue episodes without branch preview?

This is an **information/representation court**, not a deployable routing mechanism court. Passing V10 can authorize only a new DEVELOPMENT mechanism design using a preregistered representation view. It cannot authorize evaluation lineage, confirmatory execution, challenge, promotion, or a neural-success claim.

## 2. Non-reuse and lineage boundary

V10 uses no V8/V9 episodes, roots, fitted models, receipts, artifacts, labels, embeddings, thresholds, or classifiers. V9 is rationale only.

All V10 generated episodes use `rng_stream="augmentation"`. No `evaluation` RNG stream is permitted. Replicate lineages `60000..60032` are forbidden. Confirmatory examples and targets are forbidden. Challenge materialization and promotion are forbidden.

Root prefix:

`20260912-exp279-v10-cric-dev`

Training budgets:

`{60, 120}`

Canonical roots per budget:

`{0, 1, 2, 3}`

For budget `B` and canonical index `C`:

- training root: `20260912-exp279-v10-cric-dev/train-B/canonical-C/train`
- fit roots: `.../fit-0`, `.../fit-1`
- heldout roots: `.../heldout-0`, `.../heldout-1`

Each fit or heldout root contains exactly **128 generator replicates × batch 8 = 1,024 episodes**.

Per canonical root:

- fit episodes: 2,048
- heldout episodes: 2,048

Per budget:

- fit episodes: 8,192
- heldout episodes: 8,192

Across both budgets:

- fit episodes: 16,384
- heldout episodes: 16,384

No sample escalation is authorized after data are visible.

## 3. Frozen canonical model geometry

The canonical hybrid is trained from scratch independently for each `(budget, canonical index)` using existing EXP-279 matched-resource machinery.

Frozen model/world contract:

- `d_model = 64`
- `hidden_size = 48`
- `target_parameters = 500000`
- `timesteps = 4`
- `variables = 6`
- `constraints = 3`
- `batch_size = 8`
- `noise_std = 0.05`
- canonical optimizer `lr = 0.002`
- canonical optimizer `weight_decay = 0.0`
- existing hybrid route threshold `0.5`
- training stream: augmentation only

Matched-resource audit must close for parameter match, functional parameter match, active functional parameter match, optimizer-visible parameter match, reclaimed-parameter assignment, and compute-budget closure.

After canonical training, the model is frozen. Its full functional-state digest must remain unchanged during all fit and heldout representation/outcome collection.

## 4. Counterfactual action labels

For each episode, compute exact STOP and forced-BRANCH outcomes using the frozen hybrid and the existing exact-solution semantics: an episode succeeds iff every variable argmax equals its target.

Outcome partition:

- `RESCUE`: STOP fails and forced BRANCH succeeds
- `HARM`: STOP succeeds and forced BRANCH fails
- `BOTH_SUCCESS`: both succeed
- `BOTH_FAILURE`: both fail

The diagnostic action label is binary and frozen:

- `BRANCH = 1` **iff** outcome is `RESCUE`
- `STOP = 0` for `HARM`, `BOTH_SUCCESS`, and `BOTH_FAILURE`

Rationale: BRANCH is economically justified only when it adds an exact solution that STOP would miss. V10 does not reward BRANCH for already-solved or still-unsolved episodes.

Targets may be used to create augmentation-only fit labels and to score augmentation-only heldout outcomes. Targets are never part of a representation vector and never enter inference.

## 5. Preregistered representation views

All views are extracted without executing the branch GRU. No forced-branch logits, branch context, branch hidden state, outcome label, target, teacher prediction, or future state may enter any feature vector.

Let:

- `events = SiLU(event_projection(surface_events))`, shape `[B,4,48]`
- `variables = SiLU(variable_projection(variable_states))`, shape `[B,6,48]`
- `propagated = _propagate(variables, incidence)`, shape `[B,6,48]`
- `propagation_state = variables + propagated`
- `reclaimed = tanh(reclaimed_projection(propagation_state))`
- `stop_state = propagation_state + reclaimed`
- `routing_logits = routing_head(propagation_state).squeeze(-1)`
- `stop_logits = decision_head(stop_state)`
- `stop_verifier_logits = verifier_head(stop_state).squeeze(-1)`

Three views are frozen.

### 5.1 `V9_MEAN_BASELINE`

Dimension: 144.

`concat(mean(propagation_state, variables-axis), mean(events, time-axis), events_last - events_first)`

This is a control/reference bottleneck. It is not allowed by itself to authorize a successor mechanism.

### 5.2 `FULL_PREBRANCH_STATE`

Dimension: 480.

`concat(flatten(events), flatten(propagation_state))`

This tests whether information lost by mean pooling is present in the full branch-free recurrent/symbolic state before the routing decision.

### 5.3 `EXECUTION_FRONTIER_STATE`

Dimension: 312.

`concat(flatten(stop_state), flatten(stop_logits), stop_verifier_logits, routing_logits)`

This uses only state already computable on the STOP path before any branch recurrence. It intentionally moves the diagnostic observation point to the stop execution frontier without previewing the branch.

No view may be altered after any V10 heldout result becomes visible.

## 6. Frozen diagnostic rule: exact chunked 1-nearest-neighbor

Each representation is evaluated independently with the same non-parametric rule.

Fit data are concatenated in deterministic order:

1. fit root string lexical order
2. generator replicate ascending
3. batch index ascending

For every feature dimension, compute fit-set mean and population standard deviation (`unbiased=False`). Clamp standard deviation to a minimum of `1e-6`. Standardize fit and heldout vectors using these fit statistics only.

Prediction uses exact squared Euclidean **1-nearest-neighbor**, `k=1`.

- no learned model
- no threshold
- no class weighting
- no calibration
- no temperature
- no hyperparameter sweep
- no feature selection
- no PCA
- no prototype pruning
- no approximate nearest-neighbor search

Heldout inference is chunked only for memory. Frozen query chunk size: **256 episodes**. Chunking must not alter exact nearest-neighbor results.

Tie-breaking is deterministic: PyTorch `argmin` selects the first minimum; because fit ordering is frozen, equal-distance ties select the earliest fit episode in the ordering above.

The 1-NN lookup cost is **not charged** to policy utility. This is deliberate: V10 measures a representation-information upper bound, not deployability. Therefore a pass says “the representation contains economically useful rescue information under this fixed diagnostic rule,” not “this router is deployable.”

## 7. Heldout economic accounting

Use the existing matched hybrid compute ledger:

- `C_stop = stop_accounted_flops_per_episode`
- `C_branch = branch_accounted_flops_per_episode`

No diagnostic-router FLOPs are added.

For each heldout episode:

- predicted STOP → realized solution is exact STOP outcome, cost `C_stop`
- predicted BRANCH → realized solution is exact forced-BRANCH outcome, cost `C_branch`

For each representation/root compute:

- route fraction
- realized policy solutions
- realized policy FLOPs
- `U_policy = policy_solutions / policy_FLOPs`
- `U_stop = stop_successes / (N * C_stop)`
- `U_branch = branch_successes / (N * C_branch)`
- selected rescues
- selected harms
- selected both-success
- selected both-failure
- raw rescue prevalence
- selected rescue prevalence = selected rescues / selected routed episodes
- rescue enrichment = selected rescue prevalence / raw rescue prevalence when raw prevalence > 0

No heldout-derived threshold or calibration is permitted.

## 8. Root classifications

A representation on one canonical root is `REPRESENTATION_ECONOMICALLY_IDENTIFIABLE` iff all conditions hold:

1. `0 < route_fraction < 1`
2. `U_policy > U_stop`
3. `U_policy > U_branch`
4. `selected_rescues > selected_harms`
5. `selected_rescue_prevalence > raw_rescue_prevalence`
6. frozen provenance and scientific-boundary checks close

If conditions 2 and 3 hold but at least one of conditions 1, 4, or 5 fails, classify `REPRESENTATION_PARTIAL`.

Otherwise classify `REPRESENTATION_NOT_IDENTIFIABLE`.

All comparisons are strict; no tolerance band, confidence-interval override, or post-hoc exception is permitted.

## 9. Budget classifications

For each representation and budget, pool the four canonical heldout receipts using sufficient statistics only.

`REPRESENTATION_RECURRENTLY_IDENTIFIABLE` iff:

- all four roots are `REPRESENTATION_ECONOMICALLY_IDENTIFIABLE`
- pooled `U_policy > pooled U_stop`
- pooled `U_policy > pooled U_branch`
- pooled selected rescues > pooled selected harms
- pooled selected rescue prevalence > pooled raw rescue prevalence
- exact four-root provenance closes

If pooled utility beats both baselines but root robustness fails, classify `REPRESENTATION_INTERMITTENT`.

Otherwise classify `REPRESENTATION_NOT_IDENTIFIABLE`.

## 10. Cross-budget disposition

For each representation:

- `CROSS_BUDGET_IDENTIFIABLE` iff both train60 and train120 are `REPRESENTATION_RECURRENTLY_IDENTIFIABLE`
- `CROSS_BUDGET_INTERMITTENT` iff neither is NOT_IDENTIFIABLE and at least one is INTERMITTENT
- otherwise `CROSS_BUDGET_NOT_IDENTIFIABLE`

Overall V10 disposition is frozen in this priority order:

1. If `EXECUTION_FRONTIER_STATE == CROSS_BUDGET_IDENTIFIABLE`:
   - `decision = FRONTIER_SIGNAL_IDENTIFIED`
   - `successor_design_authorized = true`
   - `authorized_representation_view = EXECUTION_FRONTIER_STATE`
   - `authorization_scope = DESIGN_FRONTIER_ROUTER_MECHANISM_COURT_ONLY`
2. Else if `FULL_PREBRANCH_STATE == CROSS_BUDGET_IDENTIFIABLE`:
   - `decision = FULL_PREBRANCH_SIGNAL_IDENTIFIED`
   - `successor_design_authorized = true`
   - `authorized_representation_view = FULL_PREBRANCH_STATE`
   - `authorization_scope = DESIGN_PREBRANCH_ROUTER_MECHANISM_COURT_ONLY`
3. Else if `V9_MEAN_BASELINE == CROSS_BUDGET_IDENTIFIABLE`:
   - `decision = MEAN_CONTROL_ONLY_SIGNAL`
   - `successor_design_authorized = false`
   - `authorized_representation_view = NONE`
   - `authorization_scope = REVIEW_CONTROL_ANOMALY_ONLY`
4. Else:
   - `decision = REPRESENTATION_SIGNAL_NOT_ESTABLISHED`
   - `successor_design_authorized = false`
   - `authorized_representation_view = NONE`
   - `authorization_scope = DESIGN_REPRESENTATION_OBJECTIVE_RESEARCH_ONLY`

`mechanism_successor_authorized` is **always false in V10**. V10 can authorize design of a future mechanism court but cannot authorize a mechanism as successful.

## 11. Scientific boundaries

Every shard, budget, and cross receipt must state:

- `schema` for its exact V10 level
- `evidence_level = EV-E2`
- `scientific_evidence_eligible = false`
- `evaluation_rng_used = false`
- `evaluation_lineage_consumed = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `mechanism_successor_authorized = false`
- `raw_examples_exported = false`
- `raw_model_outputs_exported = false`

Only sufficient statistics, frozen configuration, root identities, digests, and classifications may be exported.

## 12. Receipt provenance and canonicalization

Receipts use canonical JSON bytes and `.sha256` sidecars. Every reducer must validate inputs before aggregation and recompute classifications from sufficient statistics rather than trusting caller-supplied decisions.

Every shard binds:

- protocol digest
- source-tree/code digest
- scientific branch head
- executed commit
- budget/canonical index
- exact training/fit/heldout roots
- model geometry and optimizer
- fit/heldout replicate counts
- representation definitions and dimensions
- nearest-neighbor rule and chunk size
- canonical model digest before/after court
- matched-resource ledger
- scientific boundary flags

Budget receipts bind exactly four canonical shard receipt digests. Cross receipt binds exactly train60 and train120 budget receipt digests.

All JSON map keys that represent budgets must be strings at construction time so serialize → parse → serialize is byte-identical.

## 13. Workflow/release discipline

Implementation proceeds test-first.

Before any V10 release marker is created, the exact pre-marker head must be GREEN for:

- V10 dedicated contract tests
- focused EXP-279 contract
- generic core Python 3.11 and 3.13
- model-smoke and audits
- frozen protocol verification
- compile
- V10 scientific-workflow safety tests
- V10 freeze guard reporting zero authoritative scientific runs

Scientific workflow geometry:

- 8 shard jobs: `{60,120} × {0,1,2,3}`
- 2 budget reducers
- 1 cross reducer

Scientific execution may begin only from a dedicated V10 release marker that is the sole current-head change from the preregistered GREEN head. No `workflow_dispatch` scientific bypass is allowed.

A separate freeze guard must preserve the first authoritative V10 scientific run and fail closed on duplicates.

After any V10 heldout result is visible:

- no code repair on the same scientific PR
- no seed/root/sample/model/feature/rule tuning
- no rerun to improve result
- packaging-only repair, if unavoidable, must consume immutable sealed receipts and must not regenerate episodes

The V10 diagnostic branch closes without merge after final disposition. `main` remains unchanged.

## 14. Interpretation limits

A V10 pass establishes only that a frozen internal representation carries rescue-identifying information under the preregistered exact 1-NN upper-bound diagnostic strongly enough to improve realized heldout utility across both training budgets and all canonical roots.

It does **not** establish:

- a deployable router
- a learned routing mechanism
- confirmatory evidence
- evaluation evidence
- challenge success
- NLM architectural superiority
- NLM neural-core success

A V10 fail establishes only that this frozen representation family did not meet the preregistered identifiability/economic robustness court. It does not prove that no representation or future architecture can expose the signal.
