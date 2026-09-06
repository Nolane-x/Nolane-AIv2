# EXP-282 Confirmatory-Open Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze EXP-282 pilot-derived sample size, analysis constants, and confirmatory lineage without consuming confirmatory data.

**Architecture:** Add one pure preparation/validation module over the existing paired EXP-282 artifact and Match Court registry, plus a CLI that verifies the frozen protocol SHA and writes a self-hashed preparation artifact. Keep sample-size planning transparent and fail closed when required n exceeds the frozen maximum.

**Tech Stack:** Python 3.11+, stdlib statistics/math, existing Nolane evidence hashing and GitHub CI, PyTorch only through upstream paired-artifact generation.

**Spec:** `docs/superpowers/specs/2026-09-06-exp282-confirmatory-open-prep-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or its SHA file.
- Preparation remains `EV-E2 / UNVERIFIED`.
- Pilot statistical n is the number of paired model-run replicates, never the number of named RNG streams.
- Require at least 32 paired pilot replicates.
- MESI = `0.03`, power = `0.90`, familywise alpha = `0.05`, min n = `32`, max n = `128`.
- Never clamp an infeasible required n down to 128.
- Never consume or reuse pilot observations as confirmatory observations.

---

### Task 1: Confirmatory preparation contract

**Files:**
- Create: `src/nolane_ai/experiments/exp282_confirmatory_prep.py`
- Test: `tests/test_exp282_confirmatory_prep.py`

**Interfaces:**
- Consumes: frozen EXP-282 dict, paired development artifact, neural-arm registry, analysis-code digest.
- Produces: `build_exp282_confirmatory_prep(...) -> dict` and `validate_exp282_confirmatory_prep(payload) -> list[str]`.

- [x] Write RED tests for n<32 rejection, feasible sample-size freeze, n>128 fail-closed behavior, registry/execution mismatch, and tamper rejection.
- [x] Verify RED fails because the module is missing.
- [x] Implement the minimum builder and validator.
- [x] Verify targeted tests pass.

### Task 2: Pilot and lineage hardening

**Files:**
- Modify: `src/nolane_ai/experiments/exp282_confirmatory_prep.py`
- Modify: `tests/test_exp282_confirmatory_prep.py`

**Interfaces:**
- Consumes: raw per-replicate recurrent/explicit metrics and derived contrasts.
- Produces: effect-vector digest, explicit planning constants, disjoint reserved confirmatory IDs.

- [x] Add RED tests for alpha drift, non-evaluation pilot lane, confirmatory/pilot overlap, inconsistent derived contrasts, and missing planning constants.
- [x] Verify each new assertion fails for the intended missing behavior.
- [x] Add exact contrast consistency checks and deterministic confirmatory-ID reservation.
- [x] Store `z_alpha`, `z_power`, formula text, and paired-effect digest.
- [x] Verify all targeted tests pass.

### Task 3: Clean executable CLI

**Files:**
- Create: `scripts/prepare_exp282_confirmatory_open.py`
- Create: `tests/test_exp282_confirmatory_prep_cli.py`

**Interfaces:**
- Consumes: verified frozen protocol + SHA, paired execution JSON, Match Court registry JSON.
- Produces: self-hashed `NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1` JSON.

- [x] Write a RED end-to-end test that first creates real paired/registry artifacts and then invokes the missing prep CLI.
- [x] Verify the CLI test fails because the script is absent.
- [x] Implement protocol verification, artifact validation, prep construction, output writing, and concise stdout summary.
- [x] Verify the real-artifact CLI test passes.

### Task 4: CI and version boundary

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all Task 1-3 tests and scripts.
- Produces: clean-checkout regression gate and package version `0.7.0`.

- [x] Add both confirmatory-prep test files to model-smoke.
- [x] Increase EXP-282 paired CI development pilot to 32 evaluation replicates.
- [x] Run prep CLI immediately after paired-artifact generation.
- [x] Bump package version to `0.7.0`.
- [x] Run full `pytest -q` and `python -m compileall -q src scripts`.

### Task 5: Remote integration

**Files:**
- No protocol changes permitted.

**Interfaces:**
- Consumes: clean local tree after Task 4 and PR #6 merge SHA.
- Produces: one-commit PR with clean GitHub CI evidence.

- [x] Create stacked branch from exact PR #6 head while GitHub CI is queued.
- [ ] Reparent the exact tree to the merged PR #6 main SHA.
- [ ] Confirm no `protocols/` files in compare.
- [ ] Open PR with EV-E2 boundary and development-pilot variance observation explicitly labeled non-confirmatory.
- [ ] Require Python 3.11, Python 3.13, and model-smoke success before merge.
