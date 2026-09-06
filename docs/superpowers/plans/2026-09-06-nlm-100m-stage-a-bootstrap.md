# NLM 100M Stage-A Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an executable exact-budget 100M NLM scaffold and freeze the first six Stage-A confirmatory gates.

**Architecture:** Keep protocol/audit code dependency-light and isolate PyTorch behind the model package. The 100M candidate mirrors the authoritative V0.16 allocation exactly; neural functionality is incremental and unearned capacity stays explicitly reserved rather than being mislabeled as implemented capability.

**Tech Stack:** Python 3.11+, stdlib JSON/hashlib/dataclasses, optional PyTorch 2.x, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-06-nlm-100m-stage-a-design.md`

## Global Constraints
- Total candidate parameters: exactly 100,000,000.
- Canonical experiment IDs: `EXP-###`.
- Evidence maturity namespace: `EV-*`.
- First gates: EXP-277, EXP-279, EXP-282, EXP-286, EXP-289, EXP-297.
- No neural capability may be marked verified by protocol or structural tests alone.
- Post-freeze challenge randomness is derived only after protocol/code/evaluator freeze.

---

### Task 1: Parameter Budget Contract
**Files:** `src/nolane_ai/model/budget.py`, `tests/test_budget.py`
- [ ] Write failing tests for exact 100M sum, unique region names and frozen support accounting.
- [ ] Implement immutable region specs and budget validation.
- [ ] Run targeted tests.

### Task 2: Stage-A Protocol Contract
**Files:** `src/nolane_ai/protocol/schema.py`, `protocols/stage_a_v1.json`, `tests/test_protocol.py`
- [ ] Write failing tests for six canonical gates, finite result states and required freeze fields.
- [ ] Implement validator and freeze the V1 protocol JSON.
- [ ] Run targeted tests.

### Task 3: Randomness + Evidence Identity
**Files:** `src/nolane_ai/protocol/seeds.py`, `src/nolane_ai/protocol/evidence.py`, `tests/test_seeds.py`, `tests/test_evidence.py`
- [ ] Write failing tests for deterministic domain separation and digest stability.
- [ ] Implement SHA-256 based stream derivation, challenge derivation and Evidence Packet helpers.
- [ ] Run targeted tests.

### Task 4: Executable 100M Scaffold
**Files:** `src/nolane_ai/model/config.py`, `src/nolane_ai/model/regions.py`, `src/nolane_ai/model/nlm.py`, `tests/test_model.py`
- [ ] Write failing tests for 100M meta instantiation and tiny CPU forward shape.
- [ ] Implement functional residual regions plus explicit capacity reserves.
- [ ] Run model tests and full suite.

### Task 5: Repository Operations
**Files:** `README.md`, `.github/workflows/ci.yml`, `scripts/verify_protocol.py`
- [ ] Add usage/audit docs and CI.
- [ ] Run `pytest` and protocol verification locally.
- [ ] Commit on a feature branch and open a PR.
