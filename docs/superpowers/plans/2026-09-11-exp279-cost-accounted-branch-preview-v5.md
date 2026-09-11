# EXP-279 V5 Cost-Accounted Branch-Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an augmentation-only court that tests whether reusable 1/2/3-step prefixes of the canonical branch GRU provide robust branch-rescue ranking value after charging preview and selector FLOPs directly to EXP-279's primary solution/FLOP utility.

**Architecture:** Train the canonical hybrid unchanged, freeze it, collect independent augmentation-only counterfactual stop/full-branch outcomes and frozen propagation/event representations, then cross-fit one linear selector per fixed preview depth. Each held-out fold uses non-deployable oracle route cardinality `k=true rescue count`, exact counterfactual outcomes, and path-dependent preview-aware FLOP accounting. A separate cross-cell module requires the same preview depth to produce strictly positive direct utility at train60 and train120.

**Tech Stack:** Python 3.11+, PyTorch, pytest, GitHub Actions, existing EXP-279 matched-arm/seed/protocol helpers.

**Spec:** `docs/superpowers/specs/2026-09-11-exp279-cost-accounted-branch-preview-v5-design.md`

## Global Constraints

- Base is `main@803fb474eca9cf57713f190e89ef17a113f335e5`.
- Protocol remains `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1` with SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Evidence remains `EV-E2 / UNVERIFIED`; `scientific_evidence_eligible=false`.
- Canonical route threshold remains `0.5`; V5 does not tune or use it as the oracle selector threshold.
- Preview families are exactly `PREFIX1_STATE_LINEAR`, `PREFIX2_STATE_LINEAR`, `PREFIX3_STATE_LINEAR`.
- Probe root is `20260911-exp279-cost-accounted-branch-preview-v5-dev::independent-augmentation-preview-v5`.
- Matrix is train `15/60/120`, probe replicates `198`, batch `8`, folds `3`, selector steps `200`, selector LR `1e-2`, selector weight decay `0`.
- Geometry is `d_model=64`, hidden `48`, target params `500000`, timesteps `4`, variables `6`, constraints `3`, noise `0.05`, canonical LR `2e-3`, canonical weight decay `0`.
- Training and probe data use only `rng_stream="augmentation"`.
- Raw scores never cross independently fitted folds and are never exported.
- `60000..60032` is never reserved or consumed by V5.
- No confirmatory/challenge/promotion state is altered.

---

### Task 1: Core preview/resume and direct-utility court

**Files:**
- Create: `tests/test_exp279_cost_accounted_branch_preview_v5.py`
- Create after RED: `src/nolane_ai/experiments/exp279_cost_accounted_branch_preview_v5.py`

**Interfaces:**
- Produces constants `SCHEMA`, `PREVIEW_FAMILIES`, `PROBE_ROOT_SUFFIX`.
- Produces `_preview_hidden`, `_resume_branch_hidden`, `_selector_costs`, `_fold_preview_direct_utility_metrics`, `_aggregate_fold_preview_direct_utility_metrics`, `_pairwise_ranking_loss`, `_fit_preview_fold`, `validate_exp279_cost_accounted_branch_preview_v5`, `run_exp279_cost_accounted_branch_preview_v5_court`.
- Later tasks depend on the final court artifact's `probes[kind].aggregate.direct_utility_improved` and artifact boundary fields.

- [ ] **Step 1: Write failing core tests**

Tests must assert:

```python
assert PREVIEW_FAMILIES == (
    "PREFIX1_STATE_LINEAR",
    "PREFIX2_STATE_LINEAR",
    "PREFIX3_STATE_LINEAR",
)
```

They must build a tiny canonical hybrid and verify:

```python
full = full_branch_hidden(arm, events)
prefix = preview_hidden(arm, events, depth=2)
resumed = resume_branch_hidden(arm, events, depth=2, prefix_hidden=prefix)
assert torch.allclose(full, resumed, atol=1e-6, rtol=1e-6)
```

They must verify exact cost arithmetic for a synthetic fold. For `N=4`, `k=1`, `C_stop=10`, `C_branch=30`, `G=5`, `C_selector=2`:

```python
expected = 3 * (10 + 5 + 2) + 1 * (30 + 2)
assert row["preview_routed_total_accounted_flops"] == expected
```

They must verify a selected harm reduces solution count, aggregate utility sums counts/FLOPs rather than averaging ratios, fit-side support is required, scores are fold-local/non-exported, evaluation boundary fields remain false, and tiny court artifacts state preview and selector FLOPs are charged.

- [ ] **Step 2: Run RED**

Run through the existing EXP-279 focused CI surface. Expected failure: import/collection fails because `exp279_cost_accounted_branch_preview_v5` does not exist, while older EXP-279 tests remain green up to collection.

- [ ] **Step 3: Implement minimal core**

Use existing helpers from `exp279_paired_runner`, `exp279_routing_worlds`, and `matched_routing_arms`. Do not modify canonical arm code.

Preview semantics:

```python
_, hidden = arm.branch_gru(events[:, :depth, :])
return hidden
```

Resume semantics:

```python
if depth == events.shape[1]:
    return prefix_hidden
_, hidden = arm.branch_gru(events[:, depth:, :], prefix_hidden)
return hidden
```

Selector input is exactly `torch.cat((p_mean, prefix_hidden[-1]), dim=-1)` and selector is one `nn.Linear(2 * H, 1)`.

Cost helper must compute:

```python
preview_gru = depth * (12 * H * H + 20 * H)
p_mean = variables * H
selector_head = 4 * H + 1
selector = p_mean + selector_head
```

Held-out fold total preview-routed cost must be:

```python
(N-k) * (C_stop + preview_gru + selector) + k * (C_branch + selector)
```

Do not double-count preview on selected episodes.

- [ ] **Step 4: Run GREEN and compile**

Require core tests plus `python scripts/verify_protocol.py` and `python -m compileall -q src scripts` to pass.

- [ ] **Step 5: Commit core GREEN**

Commit only the core module needed to satisfy the already-committed RED tests.

---

### Task 2: CLI execution surface

**Files:**
- Create: `tests/test_exp279_cost_accounted_branch_preview_v5_cli.py`
- Create after RED: `scripts/run_exp279_cost_accounted_branch_preview_v5_dev.py`

**Interfaces:**
- CLI consumes `run_exp279_cost_accounted_branch_preview_v5_court`.
- CLI outputs one JSON artifact and a compact stdout receipt.

- [ ] **Step 1: Write failing CLI tests**

Use tiny geometry and assert output contains:

```python
assert artifact["schema"] == "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-COURT-V5"
assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
assert artifact["fresh_evaluation_lineage_consumed"] is False
assert artifact["preview_cost_rule"]["preview_flops_charged_to_primary_utility"] is True
assert artifact["preview_cost_rule"]["selector_flops_charged_to_primary_utility"] is True
```

Add refuse-overwrite and canonical-protocol-digest rejection tests.

- [ ] **Step 2: Run CLI RED**

Expected: core remains green; CLI tests fail because runner script is absent.

- [ ] **Step 3: Implement runner**

Follow existing EXP-279 CLI pattern: refuse existing output before protocol work; validate protocol bytes and canonical digest; compute source-tree digest; `--tiny` only changes geometry for test execution; no evaluation flags exist.

- [ ] **Step 4: Run GREEN**

Focused core+CLI tests, protocol verifier, compile must pass.

- [ ] **Step 5: Commit CLI GREEN**

---

### Task 3: Cross-cell anti-cherry-pick classifier

**Files:**
- Create: `tests/test_exp279_cost_accounted_branch_preview_v5_cross_cell.py`
- Create after RED: `src/nolane_ai/experiments/exp279_cost_accounted_branch_preview_v5_cross_cell.py`

**Interfaces:**
- Produces `classify_exp279_cost_accounted_branch_preview_v5_cross_cell(cells)`.
- Decision cells are exactly train60/train120.

- [ ] **Step 1: Write failing classifier tests**

Tests must cover all preregistered decisions and hierarchy:

```python
ROBUST_PREFIX1_BRANCH_PREVIEW
ROBUST_PREFIX2_BRANCH_PREVIEW
ROBUST_PREFIX3_BRANCH_PREVIEW
NO_ROBUST_COST_ACCOUNTED_BRANCH_PREVIEW
INCONCLUSIVE_SUPPORT
```

They must prove mixed-family positives do not survive and boundary violations (`evaluation_targets_used`, `fresh_evaluation_lineage_*`, preview-cost flags) fail closed.

- [ ] **Step 2: Run RED**

Expected `ModuleNotFoundError` for cross-cell module while core+CLI remain green.

- [ ] **Step 3: Implement classifier**

Validate cell schema/boundaries/families. Compute robust families as those with `economically_routable=true` in both decision cells. Apply cheapest-first hierarchy. Emit artifact digest and keep both fresh-lineage fields false.

- [ ] **Step 4: Run GREEN**

Focused core+CLI+cross-cell tests, protocol verifier and compile pass.

- [ ] **Step 5: Commit cross-cell GREEN**

---

### Task 4: Freeze and release real augmentation-only matrix

**Files:**
- Create or update: `.github/workflows/exp279-cost-accounted-branch-preview-v5-ci.yml`

**Interfaces:**
- Contract job gates all data jobs.
- Matrix jobs output one artifact each for train15/60/120.
- Summary job reads only train60/120 and calls the committed cross-cell classifier.

- [ ] **Step 1: Add workflow with contract-only behavior first if needed**

Contract must run frozen protocol verification, all V5 focused tests, and compile using CPU-only PyTorch.

- [ ] **Step 2: Verify pre-data contract GREEN**

No scientific matrix job may execute if contract fails.

- [ ] **Step 3: Freeze real matrix in workflow**

Commands use exactly the Global Constraints values. Matrix is `[15, 60, 120]`. Each scientific job has `needs: contract`.

- [ ] **Step 4: Add evidence-boundary assertions and upload artifacts**

Assert no evaluation/confirmatory/fresh-lineage usage and both preview/selector cost-charged flags before upload.

- [ ] **Step 5: Add automated cross-cell summary**

Summary has `needs: [court]`, downloads train60/train120 artifacts, runs classifier, asserts schema/boundary, and uploads the cross-cell receipt.

- [ ] **Step 6: Commit workflow release**

This exact commit freezes scientific code, budget, families, and workflow before real V5 results are observed.

---

### Task 5: Interpret, verify, and dispose branch

**Files:**
- Update PR body/title only after results; do not modify scientific code after V5 matrix visibility.

- [ ] **Step 1: Record train15 descriptive result**

Never use train15 to authorize a successor.

- [ ] **Step 2: Record train60/train120 receipts**

For each family capture support, route count, rescues/harms, stop/routed solution counts, preview/selector costs, direct utility and artifact digest.

- [ ] **Step 3: Use automated cross-cell result as decision authority**

Do not manually cherry-pick a family.

- [ ] **Step 4: Audit artifact IDs and GitHub ZIP SHA256 receipts**

Ensure workflow artifacts correspond to the exact scientific head.

- [ ] **Step 5: Run final exact-head verification**

Require dedicated V5 workflow SUCCESS, `exp279-confirmatory-ci` SUCCESS, and generic `ci` SUCCESS including core 3.11, core 3.13 and model-smoke.

- [ ] **Step 6: Apply preregistered disposition**

If `NO_ROBUST_COST_ACCOUNTED_BRANCH_PREVIEW`: close PR without merge and authorize only representation-learning research under a new augmentation root.

If any `ROBUST_PREFIX*`: do not merge a deployable router and do not consume `60000..60032`; retain result only as authorization for a separately preregistered deployable preview-router seam.

If `INCONCLUSIVE_SUPPORT`: close without scientific absence claim and diagnose support generation separately.