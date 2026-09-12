# EXP-279 V7 Multi-Root Support Stability Audit — Design

## Status

Pre-data architectural design for a DEVELOPMENT-only EXP-279 support audit.

Start from clean `main@803fb474eca9cf57713f190e89ef17a113f335e5` after V6 CDD closed `INCONCLUSIVE_SUPPORT` without merge.

Frozen protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`, SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

Evidence remains `EV-E2 / UNVERIFIED`. V7 may not consume or reserve provisional fresh DEVELOPMENT evaluation lineage `60000..60032`, use evaluation targets, consume confirmatory data, materialize challenge data, claim promotion, or alter the frozen Stage-A protocol.

## Why V7 exists

V4 and V5 had closed support at train120 while their mechanisms were not economically useful. V6 changed the mechanism to Counterfactual Delta Distillation and, on its independent augmentation root, the train120 decision cell contained **zero rescue positives**, forcing the preregistered cross-cell result to `INCONCLUSIVE_SUPPORT`.

That support collapse means the next valid question is not another selector or representation search. It is whether the underlying branch-rescue event is stable enough across independently trained canonical models and independently generated augmentation probes to support any subsequent mechanism claim.

V7 therefore audits support generation only. It does not train a selector, distiller, preview head, router, threshold, or deployable mechanism.

## Scientific question

For canonical training budgets `60` and `120`, how stable is the branch-rescue event

`stop_exact_failure AND forced_branch_exact_success`

across:

1. independently seeded canonical model training roots;
2. independently seeded augmentation probe roots for each frozen model;
3. deterministic folds within each probe root?

The audit must distinguish model-root instability from probe-sampling intermittency before EXP-279 performs any further mechanism-level court.

## Fixed matrix

Decision train budgets are exactly:

- `60`
- `120`

There is no train15 cell in V7.

For each train budget, evaluate exactly:

- 4 independent canonical training roots;
- 4 independent probe roots per canonical model;
- 198 probe replicates per probe root;
- batch size `8`;
- 3 deterministic diagnostic folds per probe root.

Total model/probe combinations per train budget: `4 × 4 = 16`.

Total probe episodes per model/probe combination: `198 × 8 = 1584`.

Across both decision budgets, V7 therefore uses 8 independently named canonical roots and 32 independently named probe roots. No RNG root is shared between train60 and train120.

V7 may not expand roots, replicates, folds, or budgets after data visibility on this lineage.

## Frozen root naming

Canonical root family:

`20260912-exp279-multiroot-support-v7-dev::train::{N}::canonical::{C}`

where `N ∈ {60,120}` and `C ∈ {0,1,2,3}`.

Probe root family:

`20260912-exp279-multiroot-support-v7-dev::train::{N}::probe::{C}::{P}`

where `N ∈ {60,120}`, `C ∈ {0,1,2,3}` identifies the frozen canonical model, and `P ∈ {0,1,2,3}` identifies an independent probe root.

The inclusion of `C` in a probe root is namespace/provenance only. The full probe seed is distinct from every canonical seed and every other probe seed. Probe data use only `rng_stream="augmentation"`; no examples are reused across train budgets or root pairs.

## Canonical model boundary

Each canonical model is trained with the existing EXP-279 canonical training procedure and existing route threshold `0.5`.

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

After training, every canonical parameter is frozen for support measurement.

V7 may not update event projection, variable projection, propagation projection, reclaimed projection, branch GRU, decision head, verifier head, routing head, mix gate, or any other canonical parameter during the probe audit.

## Probe outcome measurement

For every held probe episode, compute both canonical counterfactual paths post-training:

- forced stop exact outcome;
- forced branch exact outcome.

Classify the episode into mutually exclusive categories:

- `RESCUE`: stop fails exactly and forced branch succeeds exactly;
- `HARM`: stop succeeds exactly and forced branch fails exactly;
- `BOTH_SUCCESS`: both paths succeed exactly;
- `BOTH_FAIL`: both paths fail exactly.

No learned probe or score exists in V7.

V7 may additionally report forced-stop solution rate and forced-branch solution rate descriptively, but those rates cannot change the support disposition.

## Diagnostic fold partitioning

Each probe replicate `r` is assigned deterministically:

`fold = (r // len(STRATA)) % 3`

Folds are descriptive support partitions only. There is no cross-fit training in V7.

For each `(train_budget, canonical_root, probe_root, fold)` record:

- episode count;
- rescue count;
- harm count;
- both-success count;
- both-fail count;
- rescue prevalence;
- non-rescue count.

Raw examples, raw model outputs, and per-example scores must not be exported.

## Support predicates

Define for every probe root:

- `probe_rescue_supported = rescue_count > 0`
- `probe_nonrescue_supported = nonrescue_count > 0`

Define for every diagnostic fold:

- `fold_rescue_supported = fold_rescue_count > 0`
- `fold_nonrescue_supported = fold_nonrescue_count > 0`

A probe root is `fold_support_closed` only if all three folds have both rescue and non-rescue support.

For each canonical model root:

- `canonical_total_rescues` is the sum of rescue counts across its four independent probe roots;
- `canonical_total_episodes = 4 × 1584 = 6336`;
- `canonical_has_any_rescue = canonical_total_rescues > 0`;
- `canonical_all_probe_roots_supported = all four probe roots have rescue and non-rescue support`;
- `canonical_all_folds_supported = all twelve diagnostic folds have rescue and non-rescue support`.

## Conservative prevalence estimate

V7 reports a one-sided 95% Wilson lower confidence bound for rescue prevalence at two levels:

1. each `(canonical_root, probe_root)` pair, using `n = 1584`;
2. each canonical root pooled across its four independent probe roots, using `n = 6336`.

For `x` rescues among `n` episodes, with `z = 1.6448536269514722`, define the standard one-sided Wilson lower bound:

`center = (p + z^2/(2n)) / (1 + z^2/n)`

`radius = z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)`

`wilson_lower = max(0, center - radius)`

where `p = x/n`.

The train-budget conservative prevalence floor is defined **only at canonical-root level**:

`canonical_prevalence_floor = min(pooled canonical-root Wilson lower bound over the 4 canonical roots)`.

A canonical root with zero rescues has Wilson lower bound `0` and therefore forces the floor to `0`. Zero-count probe roots are never silently dropped from the pooled canonical count.

Per-pair Wilson bounds are descriptive diagnostics; the canonical-root pooled floor is the only prevalence quantity permitted for prospective sample-size planning.

## Preregistered train-cell classifications

Each train budget (`60`, `120`) receives exactly one classification, evaluated in the following order.

### 1. `MODEL_ROOT_SUPPORT_COLLAPSE`

Emit this classification if **any canonical root has zero rescues across all four of its probe roots**:

`exists C: canonical_total_rescues(C) == 0`.

This is the strongest instability result because support disappears for an independently trained frozen canonical model even after four independent augmentation probes.

### 2. `PROBE_SUPPORT_INTERMITTENT`

If no canonical root collapses, emit this classification if **any probe root or any diagnostic fold lacks rescue support**:

`exists (C,P): rescue_count(C,P) == 0`

or

`exists (C,P,F): fold_rescue_count(C,P,F) == 0`.

Non-rescue support must also remain present. If any probe root or fold contains only rescues and no non-rescues, classify as `PROBE_SUPPORT_INTERMITTENT` because binary support is not closed.

### 3. `SUPPORT_RECURRENT`

Emit this classification only if:

- every canonical root has at least one rescue;
- all 16 canonical×probe pairs have both rescue and non-rescue support;
- all 48 diagnostic folds have both rescue and non-rescue support.

No prevalence threshold beyond positive support is introduced in V7. Wilson lower bounds are reported to size a possible future court, not to redefine V7 success post hoc.

## Cross-cell disposition

The only decision cells are train60 and train120.

The cross-cell classifier emits exactly one of:

### `MODEL_ROOT_SUPPORT_COLLAPSE`

If either decision cell is classified `MODEL_ROOT_SUPPORT_COLLAPSE`.

Consequence:

- no router/selector/representation successor is authorized;
- do not increase probe sample size merely to hide model-root collapse;
- no finite prospective sample-size recommendation is emitted for EXP-279 mechanism court design from V7;
- next research must revisit canonical training stability, branch semantics, or the definition of branch complementarity under a separately preregistered seam.

### `PROBE_SUPPORT_INTERMITTENT`

If neither cell has model-root collapse, but either cell is `PROBE_SUPPORT_INTERMITTENT`.

Consequence:

- no mechanism successor is authorized;
- V7 may compute a prospective augmentation-only sample-size recommendation using the canonical-root prevalence floor;
- the future court must use entirely new roots and remain DEVELOPMENT-only;
- `60000..60032` remains locked.

### `SUPPORT_RECURRENT`

Only if both train60 and train120 are `SUPPORT_RECURRENT`.

Consequence:

- V7 still does not authorize a selector or representation mechanism;
- V7 may authorize only the **design** of a new mechanism court whose probe sample size is preregistered from V7's canonical-root prevalence floor using entirely new roots;
- no fresh evaluation lineage is reserved or consumed.

## Prospective sample-size recommendation

This calculation is emitted only when the cross-cell result is `PROBE_SUPPORT_INTERMITTENT` or `SUPPORT_RECURRENT`. It is planning metadata for a separately preregistered future DEVELOPMENT court, not a V7 endpoint.

For each train budget:

`p_floor = canonical_prevalence_floor`.

If `p_floor <= 0`, emit `sample_size_recommendation = null`.

Otherwise choose a preregistered future support target of at least one rescue with probability `0.99` under the conservative floor:

`n_min = ceil(log(1 - 0.99) / log(1 - p_floor))`.

Cross-cell planning uses the more conservative of the train60 and train120 recommendations:

`future_probe_episodes_per_canonical_root = max(n_min_60, n_min_120)`.

This formula does not authorize data collection on V7 and does not change the V7 matrix.

## Anti-leak and anti-cherry-pick rules

V7 must assert all of the following:

- training RNG stream is augmentation only;
- probe RNG stream is augmentation only;
- all 8 canonical roots are distinct;
- all 32 probe roots are distinct;
- train60 and train120 share no RNG roots;
- evaluation RNG stream is unused;
- evaluation targets are unused;
- confirmatory examples are unused;
- external examples are unused;
- `60000..60032` is untouched;
- no selector is trained;
- no CDD distiller is trained;
- no branch preview selector is trained;
- no threshold is tuned;
- no root is replaced after visibility;
- no extra root is added after visibility;
- no train budget is added after visibility;
- train60 and train120 have equal decision authority;
- raw examples are not exported;
- support classification uses only preregistered count predicates.

Historical V4/V5/V6 results may motivate V7 but may not be pooled numerically into V7 prevalence estimates. V7 uses only its new independent roots.

## Required artifacts

For each train budget, emit one aggregate receipt containing:

- schema and evidence boundary;
- protocol ID/digest;
- exact code commit;
- train budget;
- all four canonical root IDs;
- all sixteen canonical×probe sufficient-statistics records;
- all 48 fold sufficient-statistics records;
- per-pair one-sided 95% Wilson lower bounds;
- per-canonical pooled one-sided 95% Wilson lower bounds;
- canonical prevalence floor;
- per-canonical support predicates;
- train-cell classification;
- digest of canonical/probe root map;
- `fresh_evaluation_lineage_may_be_reserved = false`;
- `fresh_evaluation_lineage_consumed = false`;
- `confirmatory_data_consumed = false`;
- `challenge_materialized = false`;
- `promotion_claimed = false`.

Cross-cell receipt must include:

- train60 receipt digest;
- train120 receipt digest;
- exact cross-cell disposition;
- successor design authorization scope;
- prospective sample-size recommendation if allowed;
- `fresh_evaluation_lineage_may_be_reserved = false` in every disposition.

Every JSON receipt is accompanied by a `.sha256` file. GitHub artifact ZIP digests must be retained in the final PR disposition.

## Engineering process

Before real V7 support data become visible:

1. core support-count/Wilson/classification tests establish RED, then GREEN;
2. CLI tests establish a separate RED, then GREEN;
3. cross-cell classifier tests establish a separate RED, then GREEN;
4. absolute frozen-identity tests prove two mutually consistent but wrong receipts fail closed;
5. dedicated V7 workflow is first committed in contract-only form;
6. contract-only workflow must be GREEN;
7. only then may the fixed `2 train budgets × 4 canonical roots × 4 probe roots` matrix be enabled;
8. scientific code, roots, budgets, support predicates, Wilson rule, and disposition logic are frozen once the first V7 result becomes visible.

## Completion and disposition

V7 is a DEVELOPMENT diagnostic, not production functionality.

- Do not merge a falsified or inconclusive mechanism claim into `main`.
- V7 does not itself establish a deployable router.
- `EV-E2 / UNVERIFIED` remains unchanged.
- `60000..60032` remains reserved and untouched in all V7 outcomes.
- Any V8 or successor seam requires a separate architectural design and preregistration.
