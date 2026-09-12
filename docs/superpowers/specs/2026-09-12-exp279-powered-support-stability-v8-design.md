# EXP-279 V8 Powered Support Stability Audit — Design

## Status

Pre-data architectural design for a DEVELOPMENT-only EXP-279 support-stability replication.

Start from clean `main@803fb474eca9cf57713f190e89ef17a113f335e5` after V7 closed `PROBE_SUPPORT_INTERMITTENT` without merge on PR #57.

Frozen protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`, SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

Evidence remains `EV-E2 / UNVERIFIED`. V8 may not consume or reserve provisional fresh DEVELOPMENT evaluation lineage `60000..60032`, use evaluation targets, consume confirmatory data, materialize challenge data, claim promotion, or alter the frozen Stage-A protocol.

## Why V8 exists

V7 measured branch-rescue support on fresh augmentation roots across four independently trained canonical models per decision budget. Train60 was `SUPPORT_RECURRENT`, while train120 was `PROBE_SUPPORT_INTERMITTENT` because one train120 canonical model retained rescue support at every probe root but three of its twelve diagnostic folds contained zero rescues.

The limiting V7 canonical-root lower bound was:

`p_V7_floor = 0.0025814897747382503`.

V7's planning formula yielded `1782` episodes per canonical root for a 99% chance of at least one rescue under that floor. That quantity is not sufficient for the actual V7/V8 closure criterion, which requires positive rescue support in every diagnostic fold across both train budgets. V8 therefore powers the experiment against the actual global support-closure criterion rather than merely repeating V7 with a larger but mismatched sample.

V8 is the final preregistered sample-size escalation for this support seam. If V8 remains intermittent or collapses under this power, EXP-279 may not respond by increasing augmentation sample size again on the same question. The next seam must revisit canonical training stability, branch semantics, or the definition of branch complementarity.

V8 still does not train a selector, representation learner, distiller, preview head, router, calibration model, or threshold.

## Scientific question

With a probe budget chosen so that the global probability of any zero-rescue diagnostic fold is at most 1% under the V7 conservative prevalence floor, does branch-rescue support recur across fresh independent canonical models and fresh augmentation probes at both train60 and train120?

The branch-rescue event remains exactly:

`stop_exact_failure AND forced_branch_exact_success`.

## Power target and sample-size derivation

V8 preserves the V7 matrix structure:

- 2 decision train budgets: `60`, `120`;
- 4 independent canonical roots per train budget;
- 4 independent probe roots per canonical model;
- 3 deterministic diagnostic folds per probe root.

The cross-cell recurrent-support criterion therefore requires support in:

`2 × 4 × 4 × 3 = 96` diagnostic folds.

Use the V7 train120 conservative canonical-root floor as the frozen planning prevalence:

`p_floor = 0.0025814897747382503`.

For `n_fold` episodes in one diagnostic fold, the probability of observing zero rescues under prevalence `p_floor` is:

`q_zero = (1 - p_floor)^n_fold`.

V8 controls the family-wise probability of at least one zero-rescue fold by the union bound:

`P(any zero-rescue fold) <= 96 × q_zero`.

Require:

`96 × (1 - p_floor)^n_fold <= 0.01`.

The minimal integer solution before batch alignment is `3548` episodes/fold. V8 rounds upward to the canonical batch-size multiple:

`n_fold = 3552` episodes/fold.

At this frozen value:

- `(1 - p_floor)^3552 ≈ 0.0001029458422983661`;
- `96 × q_zero ≈ 0.009882800860643146 < 0.01`.

Thus if every V8 fold has true rescue prevalence at least `p_floor`, the union-bound lower guarantee that all 96 folds contain at least one rescue is greater than 99%.

Because batch size is `8`, each fold receives exactly `444` probe replicates. Each probe root has three folds and therefore:

- `1332` probe replicates/probe root;
- `10656` episodes/probe root.

Each canonical model has four probe roots and therefore:

- `5328` probe replicates/canonical root;
- `42624` probe episodes/canonical root.

Each train-budget cell contains:

- `170496` probe episodes.

Across both decision budgets V8 contains:

- `340992` probe episodes.

These counts are frozen. V8 may not expand them after data visibility.

## Fixed matrix

Decision train budgets are exactly `60` and `120`. There is no train15 cell.

For each train budget, evaluate exactly:

- 4 independent canonical training roots;
- 4 independent probe roots per canonical model;
- 1332 probe replicates per probe root;
- batch size `8`;
- 3 deterministic diagnostic folds per probe root.

No extra roots, budgets, folds, or adaptive resampling are permitted after visibility.

## Frozen root naming

V8 uses a completely new namespace and shares no RNG root with V4–V7.

Canonical roots:

`20260912-exp279-powered-support-v8-dev::train::{N}::canonical::{C}`

where `N ∈ {60,120}` and `C ∈ {0,1,2,3}`.

Probe roots:

`20260912-exp279-powered-support-v8-dev::train::{N}::probe::{C}::{P}`

where `P ∈ {0,1,2,3}`.

All 8 canonical roots and all 32 probe roots are distinct. Train60 and train120 share no RNG root. Probe data use only `rng_stream="augmentation"`.

## Canonical model boundary

Each canonical model is trained with the existing EXP-279 canonical training procedure and frozen route threshold `0.5`.

Frozen geometry and optimizer settings remain:

- `d_model = 64`
- hidden size `48`
- target parameters `500000`
- timesteps `4`
- variables `6`
- constraints `3`
- batch size `8`
- noise std `0.05`
- canonical LR `0.002`
- canonical weight decay `0`

After training, every canonical parameter is frozen. V8 may not update any canonical parameter during support measurement.

## Probe outcome measurement

For every probe episode, compute both post-training counterfactual paths:

- forced stop exact outcome;
- forced branch exact outcome.

Classify each episode into exactly one category:

- `RESCUE`: stop fails exactly and forced branch succeeds exactly;
- `HARM`: stop succeeds exactly and forced branch fails exactly;
- `BOTH_SUCCESS`: both paths succeed exactly;
- `BOTH_FAIL`: both paths fail exactly.

No learned probe or score exists in V8. Raw examples, raw model outputs, logits, route scores, and per-example labels must not be exported.

## Diagnostic folds

Use the same deterministic fold rule as V7:

`fold = (r // len(STRATA)) % 3`.

With 1332 probe replicates, each fold must receive exactly 444 replicates and therefore exactly 3552 episodes.

For every `(train_budget, canonical_root, probe_root, fold)` record, report only sufficient statistics:

- episode count;
- rescue count;
- harm count;
- both-success count;
- both-fail count;
- non-rescue count;
- rescue prevalence;
- rescue/non-rescue support booleans.

## Support predicates

V8 intentionally preserves V7 support predicates so the powered replication cannot redefine success after the V7 outcome.

For each probe root:

- `probe_rescue_supported = rescue_count > 0`;
- `probe_nonrescue_supported = nonrescue_count > 0`.

For each diagnostic fold:

- `fold_rescue_supported = rescue_count > 0`;
- `fold_nonrescue_supported = nonrescue_count > 0`.

A probe root has `fold_support_closed=true` only if all three folds contain both rescue and non-rescue support.

For each canonical root:

- `canonical_total_rescues` sums all four probe roots;
- `canonical_total_episodes = 42624`;
- `canonical_has_any_rescue = canonical_total_rescues > 0`;
- `canonical_all_probe_roots_supported` requires all four probe roots to contain rescue and non-rescue support;
- `canonical_all_folds_supported` requires all twelve folds to contain rescue and non-rescue support.

## Prevalence reporting

V8 reports the same one-sided 95% Wilson lower bound as V7 at:

1. each canonical×probe pair, with `n=10656`;
2. each canonical root pooled across four probe roots, with `n=42624`.

Use `z = 1.6448536269514722` and the exact V7 Wilson formula.

The train-cell conservative prevalence floor remains:

`min(pooled canonical-root Wilson lower bound over the 4 canonical roots)`.

These Wilson estimates are descriptive for V8. They do not alter the support predicates or trigger another sample-size escalation.

## Preregistered train-cell classifications

Each train budget receives exactly one classification in this order.

### 1. `MODEL_ROOT_SUPPORT_COLLAPSE`

Emit if any canonical root has zero rescues across all four probe roots.

### 2. `PROBE_SUPPORT_INTERMITTENT`

If no canonical root collapses, emit if any probe root or any diagnostic fold lacks rescue support or lacks non-rescue support.

### 3. `SUPPORT_RECURRENT`

Emit only if every canonical root, all sixteen canonical×probe pairs, and all forty-eight diagnostic folds contain both rescue and non-rescue support.

No alternative prevalence threshold or score may override this ordering.

## Cross-cell disposition

The only decision cells are train60 and train120, with equal authority.

### `MODEL_ROOT_SUPPORT_COLLAPSE`

If either decision cell collapses.

Consequence:

- no mechanism successor authorized;
- no more sample escalation on this support seam;
- next research must revisit canonical training stability, branch semantics, or branch-complementarity definition under a separately preregistered design.

### `PROBE_SUPPORT_INTERMITTENT`

If neither cell collapses but either remains intermittent.

Consequence:

- no mechanism successor authorized;
- no more sample escalation on this support seam;
- V8 explicitly closes the hypothesis that V7 intermittency was merely an underpowered fold-support artifact at the V7 conservative floor;
- next research must investigate heterogeneity/stability of canonical branch semantics rather than increase probe count.

### `SUPPORT_RECURRENT`

Only if both train60 and train120 are recurrent.

Consequence:

- V8 still does not authorize a deployable selector/router;
- V8 authorizes only a separately preregistered mechanism-court design using entirely new augmentation roots;
- the mechanism court must inherit a probe-support budget no smaller than the V8 powered per-fold support design unless its own preregistered endpoint mathematically justifies another budget;
- `60000..60032` remains locked.

## No-further-escalation rule

V8 is the final support-sample escalation for EXP-279's current branch-rescue definition.

If cross-cell V8 is `MODEL_ROOT_SUPPORT_COLLAPSE` or `PROBE_SUPPORT_INTERMITTENT`:

- do not run V9 as “same support court with more examples”;
- do not add roots to V8;
- do not replace weak roots;
- do not change fold definitions;
- do not pool V7+V8 counts to rescue support closure;
- do not lower the closure requirement.

A continuation after negative V8 must change the scientific question, not merely the sample size.

## Anti-leak and anti-cherry-pick rules

V8 must assert:

- training RNG is augmentation only;
- probe RNG is augmentation only;
- all V8 roots are fresh and mutually distinct;
- evaluation RNG is unused;
- evaluation targets are unused;
- confirmatory examples are unused;
- external examples are unused;
- `60000..60032` is untouched;
- no selector, CDD distiller, preview selector, or calibration model is trained;
- no threshold is tuned;
- no root is replaced after visibility;
- no extra root/budget/fold/sample is added after visibility;
- train60 and train120 have equal decision authority;
- raw examples/model outputs are not exported;
- support classification uses only preregistered count predicates;
- V7 counts are used only to preregister the V8 power calculation and are not numerically pooled into V8 estimates.

## Provenance and artifacts

V8 fixes the provenance weakness exposed during V7 operational recovery.

Every shard receipt must contain:

- exact scientific branch-head commit;
- exact executed GitHub commit/merge SHA when different;
- source-tree digest;
- protocol ID/digest;
- root-map fragment digest;
- frozen geometry/optimizer/probe configuration;
- scientific-boundary booleans;
- artifact digest.

Every train-budget aggregate receipt must contain:

- exact scientific branch-head commit;
- exact executed commit;
- all four canonical root IDs;
- all sixteen pair sufficient-statistics records;
- all forty-eight fold sufficient-statistics records;
- pair and canonical Wilson bounds;
- conservative prevalence floor;
- train-cell classification;
- digest of the complete root map;
- source shard receipt digests;
- `fresh_evaluation_lineage_may_be_reserved=false`;
- `fresh_evaluation_lineage_consumed=false`;
- `confirmatory_data_consumed=false`;
- `challenge_materialized=false`;
- `promotion_claimed=false`.

The cross-cell receipt must include the two budget receipt digests, exact disposition, authorization scope, and the same locked boundaries.

Every JSON receipt must have a `.sha256` sidecar. Final disposition retains GitHub artifact IDs and ZIP SHA256 digests.

## Workflow safety

The dedicated V8 workflow must be committed first in contract-only form and complete GREEN before any powered support shard is enabled.

The scientific-release workflow must include a freeze guard that prevents later PR-path changes from silently re-running powered shards after first visibility. Any post-data engineering recovery must consume immutable source artifacts by pinned run ID and may not execute scientific shards again.

Reducer environments must install CPU-only PyTorch because the frozen V8 analysis module imports canonical EXP-279 model helpers. Reducers may not fail due to the V7 missing-`torch` operational defect.

## Engineering process

Before powered V8 data become visible:

1. core root/power/support/Wilson tests establish RED then GREEN;
2. CLI tests establish separate RED then GREEN;
3. cross-cell classifier tests establish separate RED then GREEN;
4. provenance tests require exact branch-head/executed commit fields and fail closed on mutually consistent wrong identities;
5. workflow/freeze-guard tests establish RED then GREEN;
6. dedicated V8 workflow is committed in contract-only form;
7. contract-only workflow must be GREEN;
8. only then may the fixed powered matrix be enabled;
9. scientific code, roots, budgets, sample size, support predicates, Wilson rule, and disposition logic are frozen once the first V8 result becomes visible.

## Completion and disposition

V8 is a DEVELOPMENT diagnostic, not production functionality.

- Do not merge a negative or diagnostic-only mechanism claim into `main`.
- V8 does not establish a deployable router.
- Evidence remains `EV-E2 / UNVERIFIED`.
- `60000..60032` remains reserved and untouched in all outcomes.
- If V8 is recurrent in both cells, only a new mechanism-court design is authorized.
- If V8 is intermittent/collapsed, the current support-sampling seam is closed and any successor must revisit canonical branch semantics/stability under a different scientific question.
