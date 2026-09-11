# EXP-279 V4 Cheap Pre-Branch Observability Court Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an augmentation-only diagnostic court that tests whether frozen cheap pre-branch observables can identify economically useful branch rescues before any branch-GRU execution.

**Architecture:** Reuse the valid V3 direct-counterfactual utility semantics and cross-fitting discipline, but replace V3's propagation-only probe inputs with a fixed three-family pre-branch hierarchy. Train the canonical hybrid exactly as `main`, freeze it, collect projected event features plus propagation state from an independent augmentation stream, train fold-local pairwise probes, and aggregate only solution/FLOP sufficient statistics. A separate cross-cell classifier requires the same family to be utility-positive at train=60 and train=120.

**Tech Stack:** Python 3.11+, PyTorch, pytest, GitHub Actions, existing Nolane-AIv2 EXP-279 protocol/seed/arm helpers.

**Spec:** `docs/superpowers/specs/2026-09-11-exp279-prebranch-observability-v4-design.md`

## Global Constraints

- Base stays `main@803fb474eca9cf57713f190e89ef17a113f335e5` for this diagnostic branch.
- Protocol stays `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1` with SHA256 `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Evidence stays `EV-E2 / UNVERIFIED`; `scientific_evidence_eligible=false`.
- Training and probe data use only the `augmentation` RNG stream.
- No evaluation or confirmatory examples/targets may enter V4.
- `60000..60032` must remain untouched and unconsumed.
- Canonical training, route threshold `0.5`, MESI, alpha, power, max-n and protected floor are unchanged.
- Probe features may call `event_projection`, `variable_projection`, `_propagate`, and deterministic tensor summaries, but may not call `_branch_context` or `branch_gru` for selector feature construction.
- `branch_gru` may execute only inside the existing same-weight forced-branch outcome helper used to create non-deployable training/diagnostic labels.
- Probe families are exactly `STATE_DEEPSETS`, `STATE_EVENT_MEAN`, `STATE_EVENT_RESIDUAL`.
- Pairwise ranking remains fold-local; raw probe scores must never be compared across folds or exported.
- Direct utility is total exact solutions divided by total path-dependent accounted FLOPs; no posterior break-even threshold is allowed.
- Real matrix is train `15/60/120`, probe replicates `198`, folds `3`, probe steps `200`, probe LR `1e-2`, batch `8`, geometry `64/48/4/6/3`, canonical LR `2e-3`, noise `0.05`.

---

### Task 1: Core V4 feature and direct-utility court

**Files:**
- Create: `tests/test_exp279_prebranch_observability_v4.py`
- Create: `src/nolane_ai/experiments/exp279_prebranch_observability_v4.py`
- Create: `.github/workflows/exp279-prebranch-observability-v4-ci.yml`

**Interfaces:**
- Consumes: `_build_seeded_triplet`, `_functional_state_digest`, `_hybrid_stop_decision_logits`, `_hybrid_forced_branch_decision_logits`, `_train_step`, `Exp279RoutingGenerator`, `STRATA`, `HybridRoutingArm`, `audit_matched_exp279_arm_triplet`, `derive_stream_seed`, `build_functional_optimizer`, `canonical_sha256`.
- Produces: `SCHEMA`, `PROBE_KINDS`, `PROBE_ROOT_SUFFIX`, `_prebranch_features`, `_pairwise_ranking_loss`, `_fold_direct_utility_metrics`, `_aggregate_fold_direct_utility_metrics`, `run_exp279_prebranch_observability_v4_court`.

- [ ] **Step 1: Write RED tests for fixed feature semantics and branch-GRU exclusion.**

Create tests that instantiate a tiny canonical hybrid, call `_prebranch_features`, and assert:

```python
assert set(features) == {"propagation_state", "p_mean", "e_mean", "e_delta"}
assert features["propagation_state"].shape == (batch, variables, hidden)
assert features["p_mean"].shape == (batch, hidden)
assert features["e_mean"].shape == (batch, hidden)
assert features["e_delta"].shape == (batch, hidden)
```

Monkeypatch `arm._branch_context` to raise and verify `_prebranch_features` still succeeds, proving feature construction cannot execute the branch recurrence.

- [ ] **Step 2: Write RED tests for the three exact probe families.**

Assert `PROBE_KINDS == ("STATE_DEEPSETS", "STATE_EVENT_MEAN", "STATE_EVENT_RESIDUAL")`; build each probe and check one score per episode. For `STATE_EVENT_RESIDUAL`, assert the concatenated vector semantics are exactly `p_mean`, `e_mean`, `e_delta`, `p_mean-e_mean`, `p_mean*e_mean`.

- [ ] **Step 3: Write RED tests for V3-valid direct utility arithmetic.**

Use synthetic stop/branch outcomes where the top-ranked episode is a rescue, then a harm. Assert route cardinality equals held-out rescue count, `solution_count_delta == selected_rescues - selected_harms`, routed total FLOPs are `stop_total + k*(C_branch-C_stop)`, and `direct_utility_improved` follows strict `routed_utility > stop_utility`.

- [ ] **Step 4: Write RED tests for fit-side and held-out support.**

Construct fold inputs where held-out support exists but fit-side rescue support is absent; `_fit_probe_fold` must emit `fit_support_closed=false`, `heldout_support_closed=true`, and `support_closed=false` without pretending the family passed.

- [ ] **Step 5: Write RED tiny end-to-end boundary test.**

Run `run_exp279_prebranch_observability_v4_court` with tiny geometry and assert:

```python
assert payload["schema"] == "NLM-EXP-279-PREBRANCH-OBSERVABILITY-COURT-V4"
assert payload["data_boundary"]["training_rng_stream"] == "augmentation"
assert payload["data_boundary"]["probe_rng_stream"] == "augmentation"
assert payload["data_boundary"]["evaluation_rng_stream_used"] is False
assert payload["feature_boundary"]["branch_gru_used_for_probe_features"] is False
assert payload["probe_economics"]["probe_is_deployable"] is False
assert payload["probe_economics"]["probe_inference_flops_charged_to_primary_utility"] is False
assert payload["probe_economics"]["successor_must_account_selector_flops"] is True
assert payload["fresh_evaluation_lineage_consumed"] is False
```

- [ ] **Step 6: Create focused workflow contract before production implementation.**

Workflow must install the CPU model environment, run `python scripts/verify_protocol.py`, run only the V4 unit test file, and compile `src`/`scripts`. No scientific matrix job exists yet in this step.

- [ ] **Step 7: Verify RED.**

Expected: protocol verification GREEN; pytest collection fails because `nolane_ai.experiments.exp279_prebranch_observability_v4` does not exist. Do not implement until this exact RED is observed.

- [ ] **Step 8: Implement minimal V4 core.**

Implement `_prebranch_features` as:

```python
events, variables = arm._validate_common(surface_events, variable_states)
incidence = arm._validate_incidence(incidence, variables)
propagation_state = variables + arm._propagate(variables, incidence)
p_mean = propagation_state.mean(dim=1)
e_mean = events.mean(dim=1)
e_delta = events[:, -1, :] - events[:, 0, :]
```

Never call `_branch_context` here.

Implement probe classes matching the spec exactly. Reuse V3's pairwise ranking, fold-local ROC-AUC/AP, oracle `k_fold`, rescue/harm accounting, total-FLOP aggregation, canonical training loop and augmentation generator patterns. For every fold require both fit-side and held-out positive/negative support.

- [ ] **Step 9: Emit descriptive probe parameter/FLOP receipts.**

Count probe parameters exactly. Compute analytical dense-layer FLOPs from fixed input widths and emit them under each family; label them descriptive only and keep them out of primary utility.

- [ ] **Step 10: Verify GREEN and compile.**

Expected: all V4 core tests pass, frozen protocol verifier passes, `python -m compileall -q src scripts` exits zero.

- [ ] **Step 11: Commit core GREEN.**

Commit message: `feat(exp279): add V4 pre-branch observability court core`.

---

### Task 2: Fail-closed V4 CLI

**Files:**
- Create: `tests/test_exp279_prebranch_observability_v4_cli.py`
- Create: `scripts/run_exp279_prebranch_observability_v4_dev.py`
- Modify: `.github/workflows/exp279-prebranch-observability-v4-ci.yml`

**Interfaces:**
- Consumes: `run_exp279_prebranch_observability_v4_court`.
- Produces: deterministic JSON receipt writer with frozen protocol/source digest checks and refuse-overwrite behavior.

- [ ] **Step 1: Write CLI RED tests first.**

Tests must assert a tiny run writes an augmentation-only V4 artifact and matching stdout digest, existing output is rejected before protocol work, and a byte-modified protocol plus matching local `.sha256` is still rejected by `require_canonical_stage_a_v1_digest`.

- [ ] **Step 2: Add CLI tests to focused workflow and verify RED.**

Expected: V4 core tests remain GREEN; CLI tests fail because `scripts/run_exp279_prebranch_observability_v4_dev.py` is absent.

- [ ] **Step 3: Implement minimal CLI.**

Follow the V3 runner pattern. Defaults must exactly match the spec. `--tiny` changes only geometry to `d_model=8`, `hidden_size=6`, `target_parameters=5000`; it does not relax scientific boundary fields.

- [ ] **Step 4: Verify CLI GREEN.**

Expected: core + CLI tests pass, protocol and compile pass.

- [ ] **Step 5: Commit CLI GREEN.**

Commit message: `feat(exp279): add V4 DEVELOPMENT CLI`.

---

### Task 3: Cross-cell anti-cherry-pick classifier

**Files:**
- Create: `tests/test_exp279_prebranch_observability_v4_cross_cell.py`
- Create: `src/nolane_ai/experiments/exp279_prebranch_observability_v4_cross_cell.py`
- Modify: `.github/workflows/exp279-prebranch-observability-v4-ci.yml`

**Interfaces:**
- Consumes: V4 cell artifacts for train=60 and train=120.
- Produces: `classify_exp279_prebranch_observability_v4_cross_cell(cells: dict[int, dict[str, Any]]) -> dict[str, Any]`.

- [ ] **Step 1: Write classifier RED tests.**

Cover each ordered decision:

```text
ROBUST_STATE_ONLY_PREBRANCH_SIGNAL
ROBUST_EVENT_MEAN_PREBRANCH_SIGNAL
ROBUST_EVENT_RESIDUAL_PREBRANCH_SIGNAL
NO_ROBUST_PREBRANCH_OBSERVABILITY
INCONCLUSIVE_SUPPORT
```

Also test mixed-family wins across 60/120 are rejected, missing decision cells raise, evaluation-boundary drift raises, and even robust outcomes keep `fresh_evaluation_lineage_may_be_reserved=false` and `fresh_evaluation_lineage_consumed=false`.

- [ ] **Step 2: Add cross-cell test to workflow and verify RED.**

Expected: core/CLI GREEN; collection fails because cross-cell module is absent.

- [ ] **Step 3: Implement minimal ordered classifier.**

Require train=60 and 120, validate schema/evidence/data boundary/probe family completeness, require every probe aggregate to have `support_closed=true`, compute families positive in both cells, and choose the first valid decision in complexity order: state-only, event-mean, event-residual, none.

- [ ] **Step 4: Verify cross-cell GREEN.**

Expected: all focused tests pass, protocol and compile pass.

- [ ] **Step 5: Commit cross-cell GREEN.**

Commit message: `feat(exp279): add V4 cross-cell robustness gate`.

---

### Task 4: Release frozen augmentation-only DEVELOPMENT matrix

**Files:**
- Modify: `.github/workflows/exp279-prebranch-observability-v4-ci.yml`
- Modify: PR #54 body to record preregistration and later receipts.

**Interfaces:**
- Consumes: V4 CLI and cross-cell classifier.
- Produces: three cell artifacts plus one automated cross-cell decision artifact.

- [ ] **Step 1: Freeze matrix workflow before observing V4 data.**

Add `development-court` with `needs: contract`, matrix `train_replicates: [15, 60, 120]`, exact root `20260911-exp279-prebranch-observability-v4-dev`, and exact frozen parameters from the spec. Upload JSON plus SHA256 for each cell.

Add `cross-cell-decision` with `needs: development-court`; download artifacts, read only train=60/120 for decision, execute `classify_exp279_prebranch_observability_v4_cross_cell`, write/hash/upload the decision receipt.

- [ ] **Step 2: Confirm workflow diff contains no evaluation lineage or evaluation RNG use.**

Search/diff must show no `60000`, `eval-start`, `evaluation_rng_stream`, or evaluation target consumption path in the runner/workflow.

- [ ] **Step 3: Commit matrix release.**

Commit message: `ci(exp279): release frozen V4 augmentation observability court`.

- [ ] **Step 4: Wait for automated matrix and cross-cell jobs to finish; do not change scientific code after data become visible.**

Record run ID, job conclusions, artifact IDs, artifact digests, JSON SHA256 and GitHub ZIP SHA256.

- [ ] **Step 5: Interpret only the preregistered cross-cell artifact.**

Train=15 is descriptive. The authoritative V4 disposition is the automated train=60/train=120 same-family classifier. Never promote an isolated cell win.

---

### Task 5: Exact-head verification and branch disposition

**Files:**
- Modify: PR #54 body with final outcome and evidence receipts.
- No production changes after scientific observations unless an engineering defect invalidates the run; any methodological redesign starts a new branch/root.

**Interfaces:**
- Consumes: exact final branch head and V4 artifacts.
- Produces: auditable merge/close disposition.

- [ ] **Step 1: Run/fetch fresh exact-head workflow evidence.**

Require all of:

```text
exp279-prebranch-observability-v4-ci: SUCCESS
exp279-confirmatory-ci: SUCCESS
ci core Python 3.11: SUCCESS
ci core Python 3.13: SUCCESS
ci model-smoke: SUCCESS
```

- [ ] **Step 2: Audit artifact integrity.**

Download all V4 artifacts, recompute JSON SHA256, compare with `.sha256`, and compare ZIP hashes with GitHub artifact digests.

- [ ] **Step 3: Audit scientific boundary.**

Confirm protocol ID/digest unchanged; all cell and cross-cell receipts retain `EV-E2 / UNVERIFIED`, no evaluation/confirmatory/challenge consumption, no promotion, and `60000..60032` untouched.

- [ ] **Step 4: Self-review diff.**

Check feature extraction never calls branch GRU, no extra probe family/statistic was added post-data, probe FLOPs are descriptive not silently folded into canonical utility, and same-family 60+120 rule is enforced by code rather than prose.

- [ ] **Step 5: Dispose branch from preregistered result.**

If decision is any `ROBUST_*`, merge only if the diagnostic court itself is reusable and engineering-clean; the result authorizes a separate deployable-selector design, not fresh lineage. If `NO_ROBUST_PREBRANCH_OBSERVABILITY`, close without merge as negative DEVELOPMENT evidence unless the reusable court is deliberately judged worth merging independent of the negative result. If `INCONCLUSIVE_SUPPORT`, close or retain draft without interpreting absence of signal.

- [ ] **Step 6: Record exact final head and all workflow/artifact receipts in PR #54.**

No completion claim until Step 1-5 have fresh evidence.