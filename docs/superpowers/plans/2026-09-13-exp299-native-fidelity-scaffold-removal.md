# EXP-299 Native Fidelity Scaffold-Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the frozen EXP-299 DEVELOPMENT court that tests whether fit-time exact-court teacher supervision leaves a held-out cross-domain neural fidelity advantage after all court scaffolding is removed from inference.

**Architecture:** Reuse the sealed-predata EXP-298 deterministic worlds and `BehavioralFidelityCourt`, but introduce a leakage-audited structural encoder and a matched two-arm neural classifier. Both arms consume identical structural pair features and primary labels; only the teacher arm receives a fixed auxiliary court-derived loss during fit. A separate paired runner performs deterministic fit/eval partitioning, freezes both models before held-out evaluation, scores neural predictions before evaluator-only court truth is consulted, emits canonical root receipts, and a cross reducer requires 4/4 canonical roots for successor-design authority.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, GitHub Actions, repository canonical SHA-256/evidence utilities.

**Spec:** `docs/superpowers/specs/2026-09-13-exp299-native-fidelity-scaffold-removal-design.md`

## Global Constraints

- Base implementation substrate is EXP-298 pre-marker head `68e1205fd56e5b1acb8f02cea56fc9af1a881734`; do not mutate the sealed EXP-298 branch.
- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- Evidence remains DEVELOPMENT / EV-E2; every protected non-claim flag stays false.
- Domains are exactly `code_invariant`, `causal_diagnosis`, `grounded_language_ambiguity`.
- Structural feature width is 128; `d_model=64`, `hidden_size=64`, `target_parameters=500000`.
- Fit/evaluation replicates per domain/root are exactly 96/32; 16 candidates per replicate.
- AdamW: lr `3e-4`, weight decay `1e-4`, one fit pass, batch 32, grad clip 1.0.
- Primary loss weight 1.0; teacher auxiliary loss weight teacher/control = 0.25/0.0.
- Held-out authority threshold is exactly 0.5.
- Per-domain gates: teacher BA >=0.70, teacher-control BA gain >=0.10, wrong-authority <=0.05, faithful-rejection <=0.10.
- No held-out receipt, witness, label, stratum, or evaluator truth may enter model causal inputs.
- No post-result threshold/representation/optimizer/sample/root/domain tuning or same-hypothesis rerun.

---

### Task 1: Structural encoder and leakage contract

**Files:**
- Create: `tests/test_exp299_structural_encoding.py`
- Create: `src/nolane_ai/experiments/exp299_structural_encoding.py`

**Interfaces:**
- Produces: `FEATURE_WIDTH = 128`.
- Produces: `encode_exp299_problem(domain_id: str, problem: object) -> tuple[float, ...]`.
- Produces: `encode_exp299_pair(domain_id: str, source: object, candidate: object) -> tuple[float, ...]`, length 512 (`source`, `candidate`, delta, abs-delta).
- Produces: `structural_encoder_contract() -> dict[str, object]` with schema/version/width and explicit evaluator-field exclusions.

- [ ] **Step 1: Write failing tests** asserting deterministic 128-wide domain encodings; faithful source/candidate equality; every frozen wrong candidate changes the pair representation; no API accepts `is_faithful`, `stratum`, receipt, or witness; language AST never truncates under V1 generators.
- [ ] **Step 2: Run focused test**: `pytest -q tests/test_exp299_structural_encoding.py`; expected RED because module does not exist.
- [ ] **Step 3: Implement minimal canonical encoders** with explicit finite symbol/operator tables and deterministic zero padding; reject overflow instead of truncating.
- [ ] **Step 4: Run focused test**; expected GREEN.
- [ ] **Step 5: Commit** `feat(exp299): add leakage-safe structural encoding`.

### Task 2: Matched native arms and teacher-only auxiliary supervision

**Files:**
- Create: `tests/test_exp299_native_arms.py`
- Create: `src/nolane_ai/experiments/matched_native_fidelity_arms.py`

**Interfaces:**
- Produces: `NativeFidelityDecision` with `authority_granted`, `authority_score`, `fidelity_score`, `teacher_prediction`, `neural_accounted_flops`.
- Produces: `build_matched_native_fidelity_arms(...) -> tuple[BinarySupervisionControlArm, CourtTeacherNativeArm]`.
- Produces: `audit_matched_native_fidelity_arms(control, teacher) -> dict[str, object]`.
- Both arms expose `forward_pair(pair_features: torch.Tensor)` and native `decide(pair_features, executable=True, threshold=0.5)`; native API has no receipt/truth/stratum argument.

- [ ] **Step 1: Write failing tests** for byte-identical initialization, parameter/forward-FLOP match, 512-input structural pair path, auxiliary-head presence on both arms, neural-only authority threshold, and absence of receipt/court arguments from native `decide`.
- [ ] **Step 2: Run focused test**; expected RED on missing module.
- [ ] **Step 3: Implement minimal matched network**: shared geometry, structural projection, comparison core, fidelity/authority heads, architecturally matched 7-value teacher head, exact target parameter budget, scalar-safe state audit.
- [ ] **Step 4: Run focused test**; expected GREEN.
- [ ] **Step 5: Commit** `feat(exp299): add matched native fidelity arms`.

### Task 3: Fit/evaluation partitioning and paired root runner

**Files:**
- Create: `tests/test_exp299_native_runner.py`
- Create: `src/nolane_ai/experiments/exp299_native_runner.py`

**Interfaces:**
- Produces: `run_exp299_root(...) -> dict[str, object]`.
- Produces: `validate_exp299_root(payload: dict[str, object], *, reconstruct: bool = False) -> list[str]`.
- Produces canonical root schema `NLM-EXP-299-NATIVE-FIDELITY-ROOT-V1`.

- [ ] **Step 1: Write failing tests** using tiny geometry (`fit_replicates=2`, `eval_replicates=2`) proving deterministic non-overlapping fit/eval identities, matched primary supervision, teacher/control auxiliary weights 0.25/0.0, no held-out truth before prediction, no model mutation during evaluation, exact metric reconstruction, false evidence flags, and fail-closed tamper behavior.
- [ ] **Step 2: Run focused test**; expected RED on missing runner.
- [ ] **Step 3: Implement deterministic partition generation** with distinct seed namespaces `fit_environment` and `eval_environment`; evaluate fit court only for targets; train both arms in lockstep with matched batches/optimizer geometry.
- [ ] **Step 4: Implement held-out pass** that computes structural features, obtains both predictions first, then adjudicates evaluator truth and appends raw rows.
- [ ] **Step 5: Implement metrics/root decision and canonical receipt** with scalar-safe `tensor_raw_bytes` state digests, partition/encoder/model/cost audits, internal `artifact_digest`, and validator structural checks.
- [ ] **Step 6: Run focused test**; expected GREEN.
- [ ] **Step 7: Commit** `feat(exp299): add native scaffold-removal root court`.

### Task 4: Cross reducer and cross-process portability

**Files:**
- Create: `tests/test_exp299_cross_reducer.py`
- Create: `tests/test_exp299_portability.py`
- Create: `src/nolane_ai/experiments/exp299_cross_reducer.py`

**Interfaces:**
- Produces: `classify_exp299_cross(roots: list[dict[str, object]], ...) -> dict[str, object]`.
- Produces cross schema `NLM-EXP-299-NATIVE-FIDELITY-CROSS-V1`.

- [ ] **Step 1: Write failing reducer tests** for exact roots `[0,1,2,3]`, matching frozen identities/config, 4/4 recurrent success, any-root failure closing authority, duplicate/missing root rejection, protected false flags, and artifact-digest tamper detection.
- [ ] **Step 2: Write failing fresh-process portability test** that serializes a tiny root to canonical JSON and invokes a fresh Python subprocess which imports the validator and validates the receipt without process-local state.
- [ ] **Step 3: Run focused tests**; expected RED.
- [ ] **Step 4: Implement reducer** using only JSON-stable string-keyed payloads and digest linkage; never use raw object identity/storage bytes.
- [ ] **Step 5: Harden root validator** so default validation checks receipt invariants/digests/raw aggregate reconstruction without re-training a stochastic model; optional deterministic reconstruction is test-only and must use canonical value serialization.
- [ ] **Step 6: Run focused tests including subprocess**; expected GREEN in producer and fresh process.
- [ ] **Step 7: Commit** `feat(exp299): add portable cross-root reducer`.

### Task 5: Frozen geometry, CLI, and workflow contracts

**Files:**
- Create: `protocols/exp299_native_fidelity_geometry_v1.json`
- Create: `protocols/exp299_native_fidelity_geometry_v1.sha256`
- Create: `scripts/run_exp299_native_fidelity_dev.py`
- Create: `tests/test_exp299_cli.py`
- Create: `tests/test_exp299_workflow_contract.py`
- Create: `tests/test_exp299_freeze_guard_contract.py`
- Create: `.github/workflows/exp299-native-fidelity-ci.yml`
- Create: `.github/workflows/exp299-native-fidelity-development.yml`
- Create: `.github/workflows/exp299-native-fidelity-freeze-guard.yml`

**Interfaces:**
- CLI writes canonical root/cross JSON plus `.sha256` sidecars with no overwrite.
- Geometry freezes all values from Global Constraints and binds Stage-A protocol digest.
- DEVELOPMENT workflow is release-marker gated and checks out exact PR head.

- [ ] **Step 1: Write failing CLI/workflow contract tests** before creating CLI/workflows.
- [ ] **Step 2: Run focused tests**; expected RED.
- [ ] **Step 3: Add canonical geometry JSON + SHA sidecar** and verifier assertions.
- [ ] **Step 4: Implement no-overwrite root/cross CLI** with exact repository/protocol/geometry/code binding.
- [ ] **Step 5: Implement dedicated CI** running all EXP-299 tests, frozen protocol/geometry checks, compileall, and fresh-process portability test.
- [ ] **Step 6: Implement DEVELOPMENT workflow**: preflight → four root jobs `[0,1,2,3]` → root validation before artifact upload → cross reducer → cross validation/upload. No marker means no DEVELOPMENT execution.
- [ ] **Step 7: Implement freeze guard** asserting zero authoritative run before release and exactly one afterward; fail/cancel duplicate execution authority.
- [ ] **Step 8: Run focused contract tests**; expected GREEN.
- [ ] **Step 9: Commit** `ci(exp299): freeze scaffold-removal development court`.

### Task 6: Exact-head pre-data gate and one-shot DEVELOPMENT release

**Files:**
- Create only after every pre-data gate is GREEN: `protocols/exp299-native-fidelity-release.lock`

- [ ] **Step 1: Open/maintain draft EXP-299 PR** against `main`; never merge the experiment runtime PR.
- [ ] **Step 2: Verify exact pre-marker head**: dedicated EXP-299 CI GREEN; generic core Python 3.11/3.13 GREEN; model-smoke GREEN; freeze guard GREEN with no DEVELOPMENT run; release marker absent.
- [ ] **Step 3: Record pre-data identity**: repository head, Stage-A digest, EXP-299 geometry digest, source-tree/code digest.
- [ ] **Step 4: Create marker-only commit** whose diff contains exactly `protocols/exp299-native-fidelity-release.lock` and the frozen identity values.
- [ ] **Step 5: Observe exactly one authoritative DEVELOPMENT run**; do not rerun failed scientific jobs unless failure is classified as pre-data-safe infrastructure failure by the frozen rules.
- [ ] **Step 6: Verify all four sealed root artifacts and cross receipt independently**, including sidecars, canonical round trips, internal artifact digests, raw metric reconstruction, state immutability, encoder leakage audit, and cross-process validator portability.
- [ ] **Step 7: Commit no post-result scientific changes**. Close experiment PR unmerged with final disposition.

### Task 7: Closure integration

**Files:**
- Create on a fresh docs-only branch from current `main`: `docs/superpowers/closures/2026-09-13-exp299-native-fidelity-scaffold-removal-closure.md` (or repository-standard equivalent).
- Modify only the current V0.16.1 program authority/matrix documentation needed to record the sealed disposition.

- [ ] **Step 1: Preserve exact result and evidence boundary**; no runtime/workflow/test/release files from the experiment branch are merged to `main` unless a separately frozen rule explicitly authorizes them.
- [ ] **Step 2: If recurrent positive**, record only `DESIGN_EXP300_INTEGRATED_COURT_ONLY`; if negative, close `H-NATIVE-01` V1 and keep EXP-300 blocked.
- [ ] **Step 3: Run docs/diff discipline checks and merge the docs-only closure PR** only when its diff matches the frozen closure scope.
