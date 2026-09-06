# NLM Stage-A Executable Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all six frozen Stage-A gates executable at EV-E2 and introduce budget-faithful neural interfaces for the four most important V0.16 reasoning regions.

**Architecture:** A dependency-light structured reasoning lab supplies oracle/mechanism headroom tests. The 100M PyTorch model remains exact-budget and gains specialized recurrent, constraint-belief, conflict and fidelity regions; unused capacity remains explicit reserve.

**Tech Stack:** Python 3.11+, stdlib dataclasses/hashlib/random, PyTorch 2.7+, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-06-stage-a-executable-reasoning-design.md`

## Global Constraints

- Do not alter `protocols/stage_a_v1.json` or its frozen digest.
- Algorithmic smoke evidence is capped at `EV-E2 / UNVERIFIED`.
- Total candidate parameters remain exactly `100,000,000`.
- Frozen parameter allocation remains exactly `10,000,000`.
- Capacity reserve is never described as implemented capability.
- Negative results are retained; do not tune smoke worlds to force the candidate to win.

---

### Task 1: Canonical structured problem and propagation

**Files:** `src/nolane_ai/reasoning/cps.py`, `src/nolane_ai/reasoning/propagation.py`, `tests/test_reasoning_core.py`

- [x] Write failing tests for exact equality-chain propagation.
- [x] Implement finite-domain variables/table constraints and generalized arc consistency.
- [x] Verify pruning and operation accounting.

### Task 2: Search, conflict headroom and local nogoods

**Files:** `src/nolane_ai/reasoning/search.py`, `src/nolane_ai/reasoning/worlds.py`, `tests/test_reasoning_core.py`

- [x] Write failing tests for branch-vs-hybrid cost, oracle conflict priority and repeated-dead-end reuse.
- [x] Implement chronological branch search, hybrid search and episode-local nogood store.
- [x] Add deterministic structure-dense and conflict worlds.

### Task 3: Belief and semantic-fidelity lanes

**Files:** `src/nolane_ai/reasoning/belief.py`, `src/nolane_ai/reasoning/fidelity.py`, `tests/test_belief.py`, `tests/test_fidelity.py`

- [x] Test normalized evidence updates and compile-valid semantic traps.
- [x] Implement explicit belief state, recurrent plumbing rival and bounded exact fidelity court.

### Task 4: Six-gate paired harness and analysis

**Files:** `src/nolane_ai/experiments/harness.py`, `src/nolane_ai/experiments/analysis.py`, `src/nolane_ai/experiments/runner.py`, `tests/test_experiment_harness.py`, `tests/test_analysis.py`, `tests/test_runner.py`

- [x] Execute all six frozen gate IDs on domain-separated environment seeds.
- [x] Preserve raw per-replicate arm metrics.
- [x] Add deterministic paired effects/bootstrap intervals.
- [x] Refuse protocol digest drift and cap output at EV-E2/UNVERIFIED.

### Task 5: Specialized 100M neural regions

**Files:** `src/nolane_ai/model/stage_a_regions.py`, `src/nolane_ai/model/regions.py`, `src/nolane_ai/model/nlm.py`, `tests/test_stage_a_neural_regions.py`

- [x] Replace four generic region paths with specialized recurrent/CBRF/conflict/fidelity computation.
- [x] Preserve exact region budgets through explicit reserve sealing.
- [x] Verify exact 100M / 90M trainable accounting on `meta`.
- [x] Verify structured reasoning and semantic-fidelity output shapes on the tiny CPU model.

### Task 6: Model audit and Stage-A learning objective

**Files:** `src/nolane_ai/model/audit.py`, `src/nolane_ai/training/stage_a.py`, `tests/test_model_audit.py`, `tests/test_training.py`

- [x] Report functional and reserved parameters per region.
- [x] Add belief/conflict/fidelity multitask losses.
- [x] Verify reserve parameters receive no gradient through Stage-A functional paths.

### Task 7: Operational scripts and CI

**Files:** `scripts/run_stage_a.py`, `scripts/audit_model.py`, `.github/workflows/ci.yml`, `README.md`

- [x] Add open-smoke evidence runner and exact-model audit command.
- [x] Keep dependency-light core CI on Python 3.11/3.13.
- [x] Add a dedicated PyTorch model-smoke job.
- [ ] Verify fresh GitHub Actions on the pull request before merge.
