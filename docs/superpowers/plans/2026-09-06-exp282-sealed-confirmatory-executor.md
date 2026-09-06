# EXP-282 Sealed Confirmatory-Open Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute only the exact authorized EXP-282 confirmatory-open replicate lineage from the exact authorized functional checkpoint, producing immutable raw paired evidence without applying the statistical decision rule.

**Architecture:** Extend the existing partial-observability generator with a backward-compatible metadata scope, then add one sealed executor module that validates Reconstruction Court, verifies and reconstructs the functional-only checkpoint, evaluates exact reserved IDs from the frozen protocol RNG root, and emits a self-hashed raw artifact. A thin CLI verifies protocol identity and refuses output overwrite; CI only runs a rehearsal fixture and never production confirmatory data.

**Tech Stack:** Python 3.11+, PyTorch, stdlib JSON/hash/pathlib, existing Nolane protocol/evidence/seed helpers.

**Spec:** `docs/superpowers/specs/2026-09-06-exp282-sealed-confirmatory-executor-design.md`

## Global Constraints

- Frozen `protocols/stage_a_v1.json` and `.sha256` bytes must not change.
- Confirmatory-open worlds use frozen protocol `rng.root_seed` and stream `evaluation` on exact reserved IDs.
- Development checkpoint root seed is reconstruction provenance only and must not become the confirmatory world root.
- No optimizer, no backward pass, no parameter update.
- Raw output stays `EV-E2 / UNVERIFIED` and is not a promotion decision.
- CI may exercise only rehearsal/synthetic authorization, never production confirmatory execution.

---

### Task 1: Backward-compatible confirmatory metadata scope

**Files:**
- Modify: `src/nolane_ai/experiments/exp282_partial_observability.py`
- Modify: `tests/test_exp282_partial_observability.py`

**Interfaces:**
- Consumes: existing `Exp282PartialObservabilityGenerator.make_batch(...)`.
- Produces: optional keyword `scope: str = "synthetic-exp282-partial-observability-development"`; metadata/digest use that exact scope.

- [x] **Step 1:** Write failing tests proving the default call preserves the development scope and a confirmatory call can request `synthetic-exp282-partial-observability-confirmatory-open` without changing tensor contents for otherwise identical seed inputs.
- [x] **Step 2:** Run `pytest -q tests/test_exp282_partial_observability.py` and verify RED because `scope` is not accepted.
- [x] **Step 3:** Implement the optional `scope` keyword, reject empty scope, and place it in metadata instead of the hard-coded development value.
- [x] **Step 4:** Re-run targeted tests and require pass.

### Task 2: Sealed checkpoint reconstruction and raw executor

**Files:**
- Create: `src/nolane_ai/experiments/exp282_confirmatory_executor.py`
- Create: `tests/test_exp282_confirmatory_executor.py`

**Interfaces:**
- Consumes: `validate_exp282_paired_development`, `validate_exp282_confirmatory_reconstruction`, `file_sha256`, `canonical_sha256`, `derive_stream_seed`, `build_matched_belief_arm_pair`, `Exp282PartialObservabilityGenerator`.
- Produces: `execute_exp282_confirmatory_open(*, protocol, protocol_digest, paired_execution_artifact, reconstruction_authorization, checkpoint_path, executor_code_digest) -> dict[str, Any]` and `validate_exp282_confirmatory_open_raw(payload) -> list[str]`.

- [x] **Step 1:** Write failing tests covering valid exact checkpoint/reconstruction; checkpoint SHA mismatch; reserve tensor rejection; reconstruction-contract digest mismatch; protocol-root vs development-root separation; exact reserved ID order/count; no model mutation; tamper + rehash detection; raw artifact remains `EV-E2 / UNVERIFIED`.
- [x] **Step 2:** Run `pytest -q tests/test_exp282_confirmatory_executor.py` and verify RED because the executor module does not exist.
- [x] **Step 3:** Implement checkpoint preflight before data materialization: frozen protocol status/EXP-282 contract, paired artifact validator, reconstruction validator, checkpoint SHA/schema/state policy, no reserve keys, exact execution-contract digest/equality.
- [x] **Step 4:** Implement reconstruction from authorized arm geometry; load functional states with `strict=False`; require missing keys are exactly reserve keys and no unexpected keys; call `.eval()` and never create an optimizer.
- [x] **Step 5:** Implement exact reserved execution using `Exp282PartialObservabilityGenerator(root_seed=protocol["rng"]["root_seed"])`, confirmatory scope, `rng_stream="evaluation"`, exact world geometry, and ordered reserved IDs under `torch.no_grad()`.
- [x] **Step 6:** Implement raw artifact/validator with exact lineage, per-replicate raw metrics/seeds/digests, `CONFIRMATORY_OPEN_EXECUTED_UNANALYZED`, data-consumed state, and self-hash. Validator independently checks exact count/order/uniqueness, metric contrast arithmetic, protocol root, status, and lineage digests.
- [x] **Step 7:** Run targeted tests and require pass.

### Task 3: CLI fail-closed execution surface

**Files:**
- Create: `scripts/run_exp282_confirmatory_open.py`
- Create: `tests/test_exp282_confirmatory_executor_cli.py`

**Interfaces:**
- Consumes JSON paired/reconstruction artifacts, exact checkpoint file, frozen protocol/digest, output path.
- Produces one raw confirmatory-open JSON artifact and concise stdout summary.

- [x] **Step 1:** Write failing CLI tests for successful rehearsal output, refusing pre-existing output, and refusing checkpoint SHA mismatch.
- [x] **Step 2:** Run the CLI test and verify RED because the script does not exist.
- [x] **Step 3:** Implement protocol schema+SHA verification, JSON loading, source-tree digest, no-overwrite output rule, executor call, canonical JSON write, and summary containing schema/status/artifact digest/replicate count only.
- [x] **Step 4:** Re-run CLI tests and require pass.

### Task 4: CI and evidence-boundary integration

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`
- Test: full repository tests

**Interfaces:**
- CI must execute executor unit/CLI tests but must not run production confirmatory-open data.

- [x] **Step 1:** Add executor unit/CLI tests to model-smoke test list.
- [x] **Step 2:** Do not add a CI command that consumes the repository's real confirmatory reserved lineage; unit/CLI tests use generated fixtures only.
- [x] **Step 3:** Bump package version one minor development step and keep package metadata/version aligned.
- [ ] **Step 4:** Run full clean CI on the final squash head and require zero failures.
- [ ] **Step 5:** Confirm compileall exits 0 on the final squash head through CI.
- [x] **Step 6:** Compare branch against base and require zero `protocols/` changes.
- [ ] **Step 7:** Merge only on the exact green squash head.
