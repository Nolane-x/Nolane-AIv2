# EXP-282 Match Court Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resource-matched, provenance-bound EXP-282 paired neural development lane without weakening the frozen confirmatory gate.

**Architecture:** Use two experiment-local arms with identical parameterized primitive topology but different cross-time state semantics. Add analytical compute accounting, deterministic partial-observability worlds, paired training/evaluation, and Match Court evidence transitions.

**Tech Stack:** Python 3.11+, PyTorch, pytest, SHA-256 canonical artifact digests.

**Spec:** `docs/superpowers/specs/2026-09-06-exp282-match-court-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- All new artifacts remain `EV-E2 / UNVERIFIED`.
- `capacity_reserve` is non-functional and excluded from optimizer parameter groups.
- Training uses `augmentation`; held-out evaluation uses `evaluation`.
- EXP-282 compute difference must be at most 5%; this implementation targets exact analytical parity.

---

### Task 1: Matched arm topology and compute ledger

**Files:**
- Modify: `src/nolane_ai/experiments/matched_belief_arms.py`
- Test: `tests/test_matched_belief_arms.py`
- Test: `tests/test_exp282_compute_ledger.py`

**Interfaces:**
- Produces: `account_matched_belief_arm_pair(..., timesteps, variables) -> dict`
- Produces: `audit_matched_belief_arm_pair(..., timesteps, variables) -> dict`

- [x] Write failing tests requiring identical primitive signatures and exact geometry scaling.
- [x] Verify RED because compute-ledger API is absent.
- [x] Replace GRU-vs-MLP execution mismatch with identical primitive topology while preserving state-semantics contrast.
- [x] Count arithmetic FLOPs and nonlinear primitive signatures separately.
- [x] Verify targeted tests pass.

### Task 2: Match Court resource-match evidence

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Test: `tests/test_neural_arm_registry.py`

**Interfaces:**
- Consumes: EXP-282 matched-pair audit.
- Produces: `PARAMETER_AND_COMPUTE_MATCH_CLOSED` while keeping `match_court=BLOCKED`.

- [x] Write failing tests for `exp282_pair_audit` ingestion.
- [x] Verify RED on unexpected argument.
- [x] Validate parameter, observation-history, primitive, and FLOP closure before accepting evidence.
- [x] Preserve the paired execution blocker.
- [x] Verify targeted tests pass.

### Task 3: Partial-observability world generator

**Files:**
- Create: `src/nolane_ai/experiments/exp282_partial_observability.py`
- Test: `tests/test_exp282_partial_observability.py`

**Interfaces:**
- Produces: deterministic `Exp282PartialObservabilityBatch` with observations, targets, visibility mask, metadata, digest.

- [x] Write failing deterministic/stream-separation tests.
- [x] Verify RED because module is absent.
- [x] Derive latent targets from `environment` and observations from `augmentation`/`evaluation`.
- [x] Retain exact visibility/noise lineage and digest.
- [x] Verify targeted tests pass.

### Task 4: Paired neural development runner

**Files:**
- Create: `src/nolane_ai/experiments/exp282_paired_runner.py`
- Test: `tests/test_exp282_paired_runner.py`

**Interfaces:**
- Produces: `run_exp282_paired_development(...) -> dict`
- Produces: `validate_exp282_paired_development(payload) -> list[str]`

- [x] Write failing tests for identical initialization, shared worlds, raw metrics, and replay determinism.
- [x] Verify RED because runner is absent.
- [x] Build byte-identical initial functional state using the named `model_init` stream.
- [x] Train both arms on shared augmentation batches with identical optimizer settings.
- [x] Evaluate on shared evaluation batches and retain raw accuracy/Brier contrasts.
- [x] Add self-hash validator and reject promotion, stream aliasing, compute mismatch, and missing lineage.
- [x] Verify targeted tests pass.

### Task 5: Match Court paired-execution transition

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Test: `tests/test_neural_arm_registry.py`

**Interfaces:**
- Consumes: validated paired execution artifact.
- Produces: `PAIRED_PARTIAL_OBSERVABILITY_DEV_READY`, still `BLOCKED` for confirmatory execution.

- [x] Write failing tests for execution-artifact ingestion and tamper rejection.
- [x] Verify RED on unexpected argument.
- [x] Bind execution digest, code digest, train/eval replicate counts, and aggregate development contrasts.
- [x] Replace the development execution blocker with the confirmatory-freeze/challenge blocker only.
- [x] Verify targeted tests pass.

### Task 6: Clean-checkout CLI and regression gate

**Files:**
- Create: `scripts/run_exp282_paired_dev.py`
- Test: `tests/test_exp282_paired_cli.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: paired execution JSON and Match Court registry JSON from a verified protocol SHA.

- [x] Write failing CLI test.
- [x] Verify RED because the script is absent.
- [x] Verify protocol schema/digest, execute paired development, audit exact-16M pilot on `meta`, and emit both artifacts.
- [x] Verify local CLI test passes.
- [x] Add the new tests and CLI smoke command to model-smoke CI.
- [ ] Run full pytest, compileall, protocol-diff audit, and remote clean CI before merge.
