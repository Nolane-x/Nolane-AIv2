# EXP-279 Stop-Path Cost Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the positive DEVELOPMENT utility observed in PR #40 is explained by the hybrid stop-path compute ledger rather than learned dynamic routing.

**Architecture:** Add a DEVELOPMENT-only post-hoc analysis module that consumes an existing `NLM-EXP-279-PAIRED-DEV-EVAL-V1` artifact. It is valid only when every evaluated hybrid route fraction is zero, then decomposes the aggregate utility ratio into an accuracy factor and an accounted-FLOP factor using the multiplicative identity `(S_h/S_p) * (C_p/C_h)`. A CLI analyzes one or more preserved DEVELOPMENT artifacts and emits a non-promotional receipt; no model weights, protocol bytes, confirmatory gates, route thresholds, evaluation IDs, or scientific decisions are changed.

**Tech Stack:** Python 3.11+, stdlib `argparse/json/math/pathlib`, pytest, GitHub Actions.

**Spec:** PR #40 negative-evidence record plus the frozen EXP-279 artifact schema on `main@934cb0a6f19a0d4ea582f0e1a036cdc9bf5348ac`.

## Global Constraints

- DEVELOPMENT-only; output evidence level remains `EV-E2` and decision remains `UNVERIFIED`.
- Never consume or materialize confirmatory challenge data.
- Require `challenge_materialized == false` and `confirmatory_data_consumed == false` in every input.
- Require route fraction exactly zero for every evaluation replicate before making a stop-path attribution.
- Preserve original artifact digests and train/evaluation lineage as provenance; do not mutate source artifacts.
- Use the artifact's charged FLOPs and verified solution rates; do not invent a new cost model.
- Treat decomposition as descriptive mechanism evidence, not confirmation of EXP-279.

---

### Task 1: Freeze the stop-path decomposition contract

**Files:**
- Create: `tests/test_exp279_stop_path_ablation.py`
- Create: `.github/workflows/exp279-stop-path-ablation-ci.yml`

**Interfaces:**
- Consumes: an EXP-279 DEVELOPMENT artifact dictionary.
- Produces: tests for `build_exp279_stop_path_ablation(payload: dict) -> dict` and `build_exp279_stop_path_sweep(payloads: list[dict]) -> dict`.

- [ ] **Step 1: Write failing unit tests** for exact-zero routing, multiplicative reconstruction, invalid routed artifacts, confirmatory-boundary rejection, and multi-artifact sweep lineage.
- [ ] **Step 2: Run the focused workflow** and verify RED because `nolane_ai.experiments.exp279_stop_path_ablation` does not exist.
- [ ] **Step 3: Commit only tests/workflow** so RED provenance is explicit.

### Task 2: Implement the descriptive decomposition

**Files:**
- Create: `src/nolane_ai/experiments/exp279_stop_path_ablation.py`

**Interfaces:**
- `build_exp279_stop_path_ablation(payload: dict[str, object]) -> dict[str, object]`
- `validate_exp279_stop_path_ablation(receipt: dict[str, object]) -> list[str]`
- `build_exp279_stop_path_sweep(payloads: list[dict[str, object]]) -> dict[str, object]`

- [ ] **Step 1: Validate input scientific boundary** (`schema`, `EV-E2`, `UNVERIFIED`, no challenge, no confirmatory consumption, non-empty evaluation rows).
- [ ] **Step 2: Require all hybrid route fractions/routed counts to be zero** and reject the attribution otherwise.
- [ ] **Step 3: Compute fixed stop-path and propagation costs** from charged per-row `accounted_flops_per_episode`; require each to be constant and positive across rows.
- [ ] **Step 4: Compute aggregate factors**: `accuracy_factor = mean_hybrid_solution_rate / mean_propagation_solution_rate`, `compute_factor = propagation_cost / hybrid_stop_cost`, and `reconstructed_utility_factor = accuracy_factor * compute_factor`.
- [ ] **Step 5: Cross-check** reconstructed factor against `mean_hybrid_utility / mean_propagation_utility` and the source artifact's `hybrid_relative_utility_gain` within strict tolerance.
- [ ] **Step 6: Classify the descriptive mechanism** without promotion. Use `COMPUTE_DOMINANT_STOP_PATH_ADVANTAGE` when total gain is positive, compute factor is >1, and accuracy factor is <=1; `COMPUTE_AND_ACCURACY_COMBINE` when both factors are >1; otherwise `NO_POSITIVE_STOP_PATH_ADVANTAGE` or `MIXED_STOP_PATH_EFFECT`.
- [ ] **Step 7: Preserve provenance**: artifact digest, training replicate count, eval start/count, protocol/code digests when present.
- [ ] **Step 8: Run focused tests and verify GREEN.**

### Task 3: Add a deterministic CLI and preserved-artifact sweep

**Files:**
- Create: `scripts/analyze_exp279_stop_path_ablation.py`
- Modify: `.github/workflows/exp279-stop-path-ablation-ci.yml`

**Interfaces:**
- CLI accepts repeated `--input PATH` and required `--output PATH`.
- One input emits a single receipt; multiple inputs emit `NLM-EXP-279-STOP-PATH-SWEEP-DEV-V1` with per-training analyses.

- [ ] **Step 1: Write CLI smoke coverage** in the focused workflow using a generated synthetic artifact fixture.
- [ ] **Step 2: Implement JSON load/analyze/write with sorted, indented deterministic output.**
- [ ] **Step 3: In GitHub Actions, download the three preserved PR #40 DEVELOPMENT artifacts from run `34571430104`** (`train=15`, `60`, `120`) using `actions/download-artifact` and analyze them together.
- [ ] **Step 4: Assert the sweep remains DEVELOPMENT-only**, has zero routing in all three inputs, never consumes confirmatory evidence, and preserves train counts `{15,60,120}`.
- [ ] **Step 5: Upload the decomposition receipt as a 90-day artifact.**

### Task 4: Review and disposition

**Files:**
- No production changes beyond Tasks 1-3.

- [ ] **Step 1: Run generic CI, focused EXP-279 contract, and stop-path ablation CI.**
- [ ] **Step 2: Inspect the preserved-artifact decomposition numbers before interpreting them.**
- [ ] **Step 3: If compute dominates, record that PR #40's positive utility is an architectural/accounting stop-path effect rather than routing evidence; if accuracy dominates or results disagree, record the actual result instead.**
- [ ] **Step 4: Keep all conclusions scoped to DEVELOPMENT and do not open confirmatory gates.**
