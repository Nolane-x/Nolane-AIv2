# EXP-279 augmentation direct-utility court V3

## Status
Preregistered DEVELOPMENT design. No V3 probe data have been generated when this plan is committed.

## Motivation
V1 was invalid because independently fitted shift-invariant fold probes were globally ranked by raw score. V2 fixed that ranking defect but used an unsupported constant posterior break-even derived only from FLOPs. EXP-279's primary metric is verified solution rate divided by accounted FLOPs, so economic viability must be evaluated from actual solution and path-dependent compute outcomes.

## Frozen scientific question
Does the frozen canonical propagation representation contain enough branch-rescue ranking signal for a fixed probe family to produce positive counterfactual verified utility when branch execution is charged exactly?

## Data boundary
- DEVELOPMENT only: EV-E2 / UNVERIFIED.
- canonical training stream: augmentation.
- probe stream: augmentation.
- new root: `20260911-exp279-augmentation-direct-utility-v3-dev`.
- independent probe suffix: `::independent-augmentation-probe-v3`.
- no evaluation RNG stream or evaluation targets.
- no confirmatory/challenge data.
- fresh DEVELOPMENT evaluation lineage `60000..60032` remains untouched.

## Frozen model/probe geometry
- canonical train replicates: 15 / 60 / 120.
- train 15 is descriptive only; robustness decision uses 60 and 120.
- probe replicates per cell: 198.
- batch size: 8 (1,584 held-out augmentation episodes per cell).
- folds: 3, assigned in blocks of all three predefined strata.
- probe families: LINEAR_MEAN, MLP_MEAN, DEEPSETS.
- probe objective: pairwise logistic rescue-over-non-rescue ranking loss.
- probe steps/fold: 200.
- probe LR: 1e-2; weight decay 0.
- d_model 64; hidden_size 48; matched target parameters 500000.
- timesteps 4; variables 6; constraints 3; noise 0.05.
- canonical LR 2e-3; weight decay 0.
- canonical route threshold remains 0.5 but is not used to define V3 diagnostic selection.

## Held-out fold procedure
For each independently fitted fold probe:
1. Freeze the canonical hybrid before any probe fitting.
2. On the held-out fold compute stop exactness and forced-branch exactness for every episode.
3. Define exact rescue as stop failure AND forced-branch success.
4. Let `k_fold` equal the number of exact rescues in that held-out fold. This is an explicitly non-deployable oracle route cardinality used only to test whether ranking information exists.
5. Rank scores only inside the held-out fold and select its top `k_fold` episodes. Raw scores are never compared across folds.
6. For selected episodes, counterfactual final exactness is forced-branch exactness. For unselected episodes it is forced-stop exactness.
7. Count selected rescues, selected harms (stop success AND branch failure), and neutral selections.
8. Charge `C_stop` to every unselected episode and `C_branch` to every selected episode using the sealed matched-arm ledger.
9. Emit sufficient statistics only: baseline/routed solution counts, baseline/routed total FLOPs, selected count, rescue/harm counts, and fold-local rank diagnostics. Raw held-out scores are not exported.

## Cross-fold economic aggregation
Aggregate by summing counts and FLOPs, never by averaging fold utility ratios:

`U_stop = sum(stop_solution_count) / sum(stop_total_flops)`

`U_routed = sum(routed_solution_count) / sum(routed_total_flops)`

`relative_gain = (U_routed - U_stop) / abs(U_stop)` when `U_stop > 0`.

A probe family is economically routable in one train cell iff:
- every held-out fold has at least one rescue and one non-rescue for probe fitting/evaluation support; and
- `U_routed > U_stop` strictly.

ROC-AUC, average precision, selected rescue precision, rescue count and harm count are descriptive diagnostics only. No AUC floor or posterior break-even threshold participates in the pass/fail rule.

## Cross-cell anti-cherry-pick rule
- `ROBUST_LINEAR_DIRECT_UTILITY`: LINEAR_MEAN is economically routable at both train 60 and train 120.
- `ROBUST_RICH_DIRECT_UTILITY`: linear is not robust, but the same richer family (MLP_MEAN or DEEPSETS) is economically routable at both train 60 and train 120.
- `NO_ROBUST_DIRECT_UTILITY`: support closes in both decision cells but no same probe family produces positive direct utility in both.
- `INCONCLUSIVE_SUPPORT`: either decision cell lacks required fold support.

Train 15 cannot authorize successor work.

Even a robust V3 result authorizes successor DESIGN only. It does not open or consume `60000..60032`, alter the canonical route threshold, or claim promotion.

## TDD order
1. Commit this plan and V3 RED unit contract before production implementation.
2. Verify frozen protocol remains GREEN and focused tests fail because the V3 module is absent.
3. Implement fold-local direct utility and court artifact until unit GREEN.
4. Add CLI tests before CLI runner; establish separate RED then GREEN.
5. Add cross-cell same-family tests before classifier; establish separate RED then GREEN.
6. Only after all pre-data gates are GREEN, release the 15/60/120 augmentation-only matrix and automated cross-cell decision.
7. Require generic CI and EXP-279 confirmatory smoke GREEN before any merge decision.
