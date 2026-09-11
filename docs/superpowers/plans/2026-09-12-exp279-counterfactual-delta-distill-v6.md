# EXP-279 V6 Counterfactual Branch-Delta Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an augmentation-only cross-fitted court that tests whether a cheap feed-forward representation distilled from canonical full-branch-minus-stop hidden-state deltas can robustly identify economically useful branch rescues at train60 and train120 after charging all CDD inference FLOPs.

**Architecture:** Train the canonical hybrid unchanged and freeze it. On each cross-fit training partition only, create detached dense branch-delta teacher vectors, fit a fixed `3H -> H -> H` CDD distiller, freeze it, then fit one linear rescue selector over `concat(p_mean, predicted_delta)`. Held-out features are branch-free; branch execution after scoring is used only for post-hoc exact outcomes in a direct solution/FLOP utility court.

**Tech Stack:** Python 3.11+, PyTorch, pytest, GitHub Actions, existing EXP-279 paired-runner/world/protocol helpers.

**Spec:** `docs/superpowers/specs/2026-09-12-exp279-counterfactual-delta-distill-v6-design.md`

## Global Constraints

- Base is `main@803fb474eca9cf57713f190e89ef17a113f335e5`.
- Protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1`, SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Evidence remains `EV-E2 / UNVERIFIED`; `scientific_evidence_eligible=false`.
- Canonical route threshold remains `0.5` and is not tuned.
- Primary family is exactly `CDD_DELTA_LINEAR`; `RAW_CHEAP_LINEAR` is descriptive only and cannot authorize a successor.
- Probe root is `20260912-exp279-counterfactual-delta-distill-v6-dev::independent-augmentation-cdd-v6`.
- Matrix is train `15/60/120`, probe replicates `198`, batch `8`, folds `3`.
- Distiller optimizer is AdamW, `200` steps, LR `2e-3`, weight decay `0`.
- Selector optimizer is AdamW, `200` steps, LR `1e-2`, weight decay `0`.
- Geometry is `d_model=64`, hidden `48`, target params `500000`, timesteps `4`, variables `6`, constraints `3`, noise `0.05`, canonical LR `2e-3`, canonical weight decay `0`.
- Canonical training and probe data use only `rng_stream="augmentation"`.
- Teacher branch hidden/state generation is fit-partition only.
- Held-out branch hidden/state is never used for features.
- Raw selector scores never cross independently fitted folds and are never exported.
- All cheap-summary, distiller and selector inference FLOPs are charged on every held-out episode.
- `60000..60032` is never reserved or consumed by V6.
- No confirmatory/challenge/promotion state is altered.

---

### Task 1: Core CDD teacher, distiller, selector, and direct-utility court

**Files:**
- Create: `tests/test_exp279_counterfactual_delta_distill_v6.py`
- Create after RED: `src/nolane_ai/experiments/exp279_counterfactual_delta_distill_v6.py`

**Interfaces:**
- Constants: `SCHEMA`, `PRIMARY_FAMILY`, `CONTROL_FAMILY`, `PROBE_ROOT_SUFFIX`.
- Core helpers: `_cheap_features`, `_stop_and_branch_states`, `_teacher_delta`, `_cdd_costs`, `_pairwise_ranking_loss`, `_fold_direct_utility_metrics`, `_aggregate_fold_direct_utility_metrics`, `_fit_cdd_fold`.
- Public functions: `validate_exp279_counterfactual_delta_distill_v6`, `run_exp279_counterfactual_delta_distill_v6_court`.
- Later tasks consume `artifact["probes"]["CDD_DELTA_LINEAR"]["aggregate"]["direct_utility_improved"]`, support fields, and boundary fields.

- [ ] **Step 1: Write failing core tests**

Tests must lock the primary/control family identities:

```python
assert PRIMARY_FAMILY == "CDD_DELTA_LINEAR"
assert CONTROL_FAMILY == "RAW_CHEAP_LINEAR"
```

Build a tiny canonical hybrid and test cheap feature construction:

```python
cheap, p_mean = _cheap_features(
    arm,
    surface_events=events,
    variable_states=variables,
    incidence=incidence,
)
assert cheap.shape == (batch, 3 * arm.hidden_size)
assert p_mean.shape == (batch, arm.hidden_size)
```

Test dense teacher semantics by computing stop and branch states through canonical helpers:

```python
stop_state, branch_state = _stop_and_branch_states(...)
teacher = _teacher_delta(stop_state, branch_state)
assert torch.allclose(teacher, (branch_state - stop_state).mean(dim=1).detach())
assert teacher.requires_grad is False
```

Test CDD inference cost exactly. For `H=4`, `V=3`, `T=5`:

```python
expected_summary = V*H + T*H + H
expected = expected_summary + (2*3*H*H + H) + 4*H + (2*H*H + H) + (4*H + 1)
assert _cdd_costs(hidden_size=H, variables=V, timesteps=T)["total_cdd_inference_flops"] == expected
```

Test direct utility arithmetic for `N=4`, `k=1`, `C_stop=10`, `C_branch=30`, `C_CDD=3`:

```python
expected = 3 * (10 + 3) + 1 * (30 + 3)
assert row["cdd_routed_total_accounted_flops"] == expected
```

Also assert:

- selected harm reduces exact solution count;
- aggregate sums counts/FLOPs instead of averaging utility ratios;
- selector fit and held-out partitions both require positive/negative rescue support;
- held-out feature records assert `branch_hidden_used_for_features=false`;
- distiller parameters are frozen before selector fitting;
- raw scores are not exported and not compared across folds;
- tiny court artifact states canonical model frozen, teacher fit-only, evaluation stream false, fresh lineage false, and CDD FLOPs charged.

- [ ] **Step 2: Run RED through existing EXP-279 focused CI**

Expected: collection/import fails because `nolane_ai.experiments.exp279_counterfactual_delta_distill_v6` does not exist. No production module is allowed before this exact RED is observed.

- [ ] **Step 3: Implement minimal core**

Reuse canonical helpers from `exp279_paired_runner` and `matched_routing_arms`; do not modify canonical arm classes.

`_cheap_features` must call canonical `_validate_common`, `_validate_incidence`, `_propagate`, then compute:

```python
propagation_state = variables + propagated
p_mean = propagation_state.mean(dim=1)
e_mean = projected_events.mean(dim=1)
e_delta = projected_events[:, -1] - projected_events[:, 0]
cheap = torch.cat((p_mean, e_mean, e_delta), dim=-1)
```

`_stop_and_branch_states` must reproduce canonical state semantics:

```python
reclaimed = torch.tanh(arm.reclaimed_projection(propagation_state))
stop_state = propagation_state + reclaimed
branch_context = arm._branch_context(projected_events).unsqueeze(1).expand(-1, variables_count, -1)
branch_state = propagation_state + torch.sigmoid(arm.mix_gate) * (branch_context + reclaimed)
```

Distiller:

```python
nn.Sequential(
    nn.Linear(3 * hidden_size, hidden_size),
    nn.SiLU(),
    nn.Linear(hidden_size, hidden_size),
)
```

Selector:

```python
nn.Linear(2 * hidden_size, 1)
```

Distiller trains only on fit partition MSE to detached teacher. Freeze all distiller parameters before selector optimization. Selector uses pairwise rescue ranking. A separate descriptive raw selector may use `cheap` directly but must not affect primary classification.

Held-out scoring must compute only cheap features, distiller prediction and selector score before post-hoc branch outcomes are materialized.

- [ ] **Step 4: Run GREEN**

Require all EXP-279 focused tests, `python scripts/verify_protocol.py`, and `python -m compileall -q src scripts` to pass on the exact core head.

- [ ] **Step 5: Commit core GREEN**

Commit only the core implementation needed by the already-committed RED tests.

---

### Task 2: CLI execution surface

**Files:**
- Create: `tests/test_exp279_counterfactual_delta_distill_v6_cli.py`
- Create after RED: `scripts/run_exp279_counterfactual_delta_distill_v6_dev.py`

**Interfaces:**
- CLI calls `run_exp279_counterfactual_delta_distill_v6_court`.
- Outputs one JSON artifact and compact stdout receipt.

- [ ] **Step 1: Write failing CLI tests**

Use tiny geometry and assert:

```python
assert artifact["schema"] == "NLM-EXP-279-COUNTERFACTUAL-DELTA-DISTILL-COURT-V6"
assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
assert artifact["data_boundary"]["heldout_branch_hidden_used_for_features"] is False
assert artifact["fresh_evaluation_lineage_consumed"] is False
assert artifact["cdd_cost_rule"]["cdd_inference_flops_charged_to_primary_utility"] is True
assert artifact["teacher_rule"]["teacher_uses_fit_partition_only"] is True
```

Add tests that output overwrite is refused before protocol work and that a byte-modified/noncanonical protocol is rejected even if accompanied by a matching local digest file.

- [ ] **Step 2: Run CLI RED**

Expected: core tests remain green; CLI tests fail because runner script is absent.

- [ ] **Step 3: Implement runner**

Follow canonical EXP-279 runner pattern:

- reject existing output first;
- validate canonical frozen protocol bytes/digest;
- compute source-tree digest;
- expose `--tiny` only for test geometry;
- expose train/probe/fold/distiller/selector hyperparameters used by workflow;
- do not expose evaluation-start, evaluation RNG, threshold tuning, confirmatory, or challenge flags.

- [ ] **Step 4: Run GREEN**

Focused core+CLI tests, frozen protocol verifier and compile must pass.

- [ ] **Step 5: Commit CLI GREEN**

---

### Task 3: Cross-cell anti-cherry-pick classifier

**Files:**
- Create: `tests/test_exp279_counterfactual_delta_distill_v6_cross_cell.py`
- Create after RED: `src/nolane_ai/experiments/exp279_counterfactual_delta_distill_v6_cross_cell.py`

**Interfaces:**
- Public function: `classify_exp279_counterfactual_delta_distill_v6_cross_cell(cells)`.
- Decision cells are exactly train60 and train120.

- [ ] **Step 1: Write failing classifier tests**

Cover exactly:

```python
ROBUST_CDD_BRANCH_COMPLEMENTARITY
NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY
INCONCLUSIVE_SUPPORT
```

Tests must prove:

- `RAW_CHEAP_LINEAR` positivity cannot authorize a successor;
- CDD must be `direct_utility_improved=true` in both train60 and train120;
- missing fit or held-out support yields `INCONCLUSIVE_SUPPORT`;
- schema/evidence/boundary mismatches fail closed;
- evaluation/confirmatory/fresh-lineage flags fail closed;
- cross-cell receipt leaves `fresh_evaluation_lineage_may_be_reserved=false` and `fresh_evaluation_lineage_consumed=false` even when robust.

- [ ] **Step 2: Run RED**

Expected `ModuleNotFoundError` for cross-cell module while core+CLI remain green.

- [ ] **Step 3: Implement classifier**

Validate real cell schema rather than synthetic aliases. Require:

```python
cell["evidence_level"] == "EV-E2"
cell["decision"] == "UNVERIFIED"
cell["canonical_training"]["replicates"] in {60, 120}
cell["probes"]["CDD_DELTA_LINEAR"]["aggregate"]["support_closed"] is True
```

Robust iff CDD aggregate direct utility is strictly positive in both decision cells. Descriptive control never enters the decision.

- [ ] **Step 4: Run GREEN**

Focused core+CLI+cross-cell tests, protocol verifier and compile pass.

- [ ] **Step 5: Commit cross-cell GREEN**

---

### Task 4: Freeze and release dedicated V6 scientific workflow

**Files:**
- Create: `.github/workflows/exp279-counterfactual-delta-distill-v6-ci.yml`

**Interfaces:**
- `contract` job gates all real data jobs.
- `court` matrix emits train15/train60/train120 artifacts.
- `cross-cell` job downloads only decision-cell artifacts and runs the committed classifier.

- [ ] **Step 1: Create contract-first workflow**

Use CPU-only PyTorch where repo patterns support it. Contract runs:

```bash
pytest -q tests/test_exp279_counterfactual_delta_distill_v6.py \
  tests/test_exp279_counterfactual_delta_distill_v6_cli.py \
  tests/test_exp279_counterfactual_delta_distill_v6_cross_cell.py
python scripts/verify_protocol.py
python -m compileall -q src scripts
```

- [ ] **Step 2: Verify pre-data contract GREEN**

No court job may run without `needs: contract`.

- [ ] **Step 3: Freeze real matrix**

Matrix is exactly `[15, 60, 120]`. Each command uses Global Constraints values and the fixed V6 root. No scientific value is supplied from an environment variable or manual workflow input.

- [ ] **Step 4: Add evidence-boundary assertions before artifact upload**

Assert canonical frozen, fit-only teacher, held-out branch hidden excluded from features, evaluation false, confirmatory false, fresh lineage false, CDD inference FLOPs charged, raw scores non-exported/non-cross-fold.

- [ ] **Step 5: Add automated cross-cell decision**

Download train60/train120 artifacts only, invoke the committed classifier, assert cross-cell boundary, and upload one cross-cell receipt.

- [ ] **Step 6: Commit scientific release head**

After this exact commit, scientific code/root/budget/loss/architecture/disposition rule are frozen. No scientific code change is allowed once any real V6 cell result becomes visible.

---

### Task 5: Interpret, audit, verify, and dispose V6

**Files:**
- Update PR body/title only after results. Do not modify scientific code after matrix visibility.

- [ ] **Step 1: Record train15 descriptive receipt**

Capture CDD/control support, route count, selected rescues/harms, stop/C DD solution counts, exact FLOPs, direct utility and artifact digest. Never let train15 authorize successor work.

- [ ] **Step 2: Record train60/train120 decision receipts**

For each decision cell capture CDD aggregate support, rescue count, selected rescues/harms, stop/routed solution counts, CDD cost per episode, stop/routed total FLOPs, utility delta, and artifact digest. Record control only as descriptive context.

- [ ] **Step 3: Use automated cross-cell receipt as sole decision authority**

Do not reinterpret an isolated cell or descriptive control as survival.

- [ ] **Step 4: Audit artifact provenance**

Verify artifact IDs correspond to the exact scientific workflow/head. Download each ZIP, recompute ZIP SHA256 and embedded JSON SHA/receipt where available, and confirm hashes match GitHub metadata/receipts.

- [ ] **Step 5: Fresh exact-head verification**

Require:

- dedicated V6 workflow SUCCESS;
- `exp279-confirmatory-ci` SUCCESS;
- generic `ci` SUCCESS including Python 3.11, Python 3.13 and model-smoke;
- frozen protocol verifier unchanged.

- [ ] **Step 6: Apply preregistered disposition**

If `ROBUST_CDD_BRANCH_COMPLEMENTARITY`: retain V6 as DEVELOPMENT authorization only; do not merge a production router and do not consume `60000..60032`. Next seam must separately preregister a deployable CDD router.

If `NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY`: close PR without merge and stop further small selector/pre-branch-feature search under EXP-279; next research must revisit routing architecture/objective or branch-complementarity definition.

If `INCONCLUSIVE_SUPPORT`: close without absence claim and diagnose support generation with a new augmentation root.
