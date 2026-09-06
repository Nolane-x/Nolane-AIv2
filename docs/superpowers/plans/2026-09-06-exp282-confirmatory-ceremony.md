# EXP-282 Confirmatory Ceremony Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-phase EXP-282 confirmatory ceremony that freezes all scientific/code/checkpoint lineage before any confirmatory observation is materialized, then executes raw confirmatory-open evidence and analysis without rewriting either artifact.

**Architecture:** Add one seal module and one ceremony execution module around the existing prep, execution authorization, reconstruction authorization, sealed executor, and Analysis Court. Add two fail-closed CLIs and model-smoke coverage. Real confirmatory data and challenge randomness remain out of scope.

**Tech Stack:** Python 3.11/3.13, PyTorch, stdlib JSON/path handling, existing Nolane evidence/protocol helpers, pytest, GitHub Actions model-smoke.

**Spec:** `docs/superpowers/specs/2026-09-06-exp282-confirmatory-ceremony-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- Canonical frozen Stage-A V1 SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- SEAL remains `EV-E2 / UNVERIFIED` with no confirmatory data or seed materialization.
- EXECUTE may materialize only reserved confirmatory-open IDs under the protocol `evaluation` stream.
- Challenge materialization remains false in every artifact.
- Analysis output may reach at most EV-E3; this PR itself is not empirical EV-E3 evidence.
- No-overwrite CLI behavior is mandatory.
- Current source-tree digest must equal the sealed code digest before execution.

---

### Task 1: Ceremony seal contract

**Files:**
- Create: `tests/test_exp282_confirmatory_ceremony.py`
- Create: `src/nolane_ai/experiments/exp282_confirmatory_ceremony.py`

**Interfaces:**
- Produces `seal_exp282_confirmatory_ceremony(...) -> dict[str, Any]`
- Produces `validate_exp282_confirmatory_ceremony_seal(payload) -> list[str]`
- Schema: `NLM-EXP-282-CONFIRMATORY-CEREMONY-SEAL-V1`

- [ ] **Step 1: Write RED tests** covering valid seal, not-ready prep rejection, tamper+rehash rejection, reserved-ID drift, code-digest drift, and challenge boundary.
- [ ] **Step 2: Run clean RED test** and preserve missing-module failure.
- [ ] **Step 3: Implement minimal seal builder/validator** by reusing existing upstream validators; do not duplicate scientific logic.
- [ ] **Step 4: Run targeted tests** until GREEN.

### Task 2: Sealed ceremony executor

**Files:**
- Modify: `tests/test_exp282_confirmatory_ceremony.py`
- Modify: `src/nolane_ai/experiments/exp282_confirmatory_ceremony.py`

**Interfaces:**
- Produces `execute_exp282_confirmatory_ceremony(...) -> dict[str, Any]`
- Result schema: `NLM-EXP-282-CONFIRMATORY-CEREMONY-RESULT-V1`
- Consumes existing `execute_exp282_confirmatory_open(...)` and `build_exp282_confirmatory_analysis(...)`.

- [ ] **Step 1: Add RED tests** proving source-tree mismatch and checkpoint mismatch abort before executor invocation, plus a synthetic successful bundle.
- [ ] **Step 2: Run RED tests** and confirm expected failures.
- [ ] **Step 3: Implement preflight order**: validate seal → canonical protocol → source-tree equality → upstream digest equality → checkpoint SHA → call sealed executor → call Analysis Court.
- [ ] **Step 4: Validate returned bundle** binds seal/raw/analysis digests and carries analysis decision without rewriting raw.
- [ ] **Step 5: Run targeted tests** to GREEN.

### Task 3: Separate SEAL and EXECUTE CLIs

**Files:**
- Create: `scripts/seal_exp282_confirmatory_ceremony.py`
- Create: `scripts/execute_exp282_confirmatory_ceremony.py`
- Create: `tests/test_exp282_confirmatory_ceremony_cli.py`

**Interfaces:**
- Seal CLI inputs: protocol, paired execution, prep, execution authorization, reconstruction authorization, output.
- Execute CLI inputs: seal, protocol, paired execution, prep, reconstruction authorization, checkpoint, raw-output, analysis-output, bundle-output.

- [ ] **Step 1: Write RED CLI tests** for successful synthetic seal, no-overwrite, forged protocol+matching digest rejection, source-tree drift rejection, and execute no-overwrite.
- [ ] **Step 2: Run RED tests** while scripts are absent.
- [ ] **Step 3: Implement SEAL CLI** using canonical protocol verification and `source_tree_digest(ROOT)`.
- [ ] **Step 4: Implement EXECUTE CLI** with all outputs preflight-checked for nonexistence before scientific execution; write raw and analysis before the final bundle only after in-memory validation succeeds.
- [ ] **Step 5: Run CLI tests** to GREEN.

### Task 4: Release/CI boundary

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

- [ ] **Step 1: Add ceremony tests to `model-smoke`** only; do not add any real confirmatory execution command to CI.
- [ ] **Step 2: Bump package version from `0.9.0` to `0.10.0`** in both version surfaces.
- [ ] **Step 3: Run full clean CI** including core 3.11, core 3.13, model-smoke, audits, development smokes, protocol verification, and compileall.
- [ ] **Step 4: Reviewer pass** checks zero protocol drift, no challenge materialization, no real confirmatory execution, and no new hidden data lane.
- [ ] **Step 5: Squash to one commit using the verified tree**, run exact-final-head CI, and merge only with expected-head SHA.

## Self-review

- Spec coverage: SEAL boundary, EXECUTE boundary, fail-closed lineage, no-overwrite, challenge prohibition, append-only raw+analysis binding, and CI boundary are all mapped to tasks.
- Placeholder scan: no TODO/TBD or unspecified implementation steps remain.
- Type consistency: both CLIs and tests use the same seal/result schemas and function names defined in Tasks 1–2.
- Scope: secure cross-host artifact transport and globally replay-proof locking remain explicitly out of scope and are not silently approximated.
