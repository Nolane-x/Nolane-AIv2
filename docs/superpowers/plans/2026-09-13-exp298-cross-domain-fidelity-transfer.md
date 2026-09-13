# EXP-298 Cross-Domain Fidelity Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot DEVELOPMENT / EV-E2 cross-domain fidelity court that tests whether the EXP-297 verification fabric transfers into bounded code invariants, causal diagnosis, and controlled grounded-language ambiguity.

**Architecture:** Three deterministic domain generators implement exact finite behavioral semantics behind one `BehavioralFidelityCourt`. Two matched neural arms differ only by receipt access; root receipts reconstruct every row and a four-root cross reducer alone can authorize design of EXP-299.

**Tech Stack:** Python 3.11/3.13, dataclasses, Protocol, PyTorch matched diagnostic arms, pytest, canonical SHA256 evidence helpers, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-exp298-cross-domain-fidelity-transfer-design.md`

## Global Constraints

- Base lineage: `main@150252b63f7268157e66024a4f9e55ad61088e35`.
- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- V1 roots `[0,1,2,3]`; 3 domains; 32 eval replicates/domain/root; 16 candidates/replicate.
- Geometry: `d_model=64`, `hidden_size=64`, `target_parameters=500000`, `max_exact_probes=4096`.
- Root primary: worst-domain BA gain; every domain must reach `>=0.10` with wrong-authority `<=0.05` and faithful-rejection `<=0.10`.
- DEVELOPMENT / EV-E2 only; all unrestricted-semantic/open-language/causal-discovery/general-code and parent-authority inheritance claims remain false.
- No post-result tuning, replacement roots, domain removal, geometry/cardinality changes, or sample escalation.

---

### Task 1: Generic Behavioral Fidelity Court

**Files:**
- Create: `src/nolane_ai/experiments/exp298_behavioral_fidelity.py`
- Test: `tests/test_exp298_behavioral_fidelity.py`

**Interfaces:**
- Produces: `BehavioralProbe`, `BehavioralOutcome`, `BehavioralFidelityReceipt`, `BehavioralSemantics`, `BehavioralFidelityCourt.adjudicate(source, candidate, semantics)`.

- [ ] Write failing tests for exact accept, directional reject, ceiling-driven inconclusive, canonical witness ordering, and digest stability.
- [ ] Run `pytest -q tests/test_exp298_behavioral_fidelity.py` and confirm RED because the module does not exist.
- [ ] Implement the minimal generic types and court. The adjudicator must consume `semantics.probes()`, reject when source/candidate outcomes differ, count probe/domain operations, and accept only after exhaustive completion.
- [ ] Run the focused test file GREEN and `python -m compileall -q src`.
- [ ] Commit `feat(exp298): add behavioral fidelity court`.

### Task 2: Code-Invariant Domain

**Files:**
- Create: `src/nolane_ai/experiments/exp298_code_worlds.py`
- Test: `tests/test_exp298_code_worlds.py`

**Interfaces:**
- Produces: `CodeInvariantProblem`, `CodeInvariantSemantics`, `generate_exp298_code_world(seed: int)` returning source + ordered 16-case candidate batch.

- [ ] Write failing tests asserting deterministic generation, exactly 16 candidates, balanced faithful/wrong support, all frozen strata represented across deterministic seeds, executable candidates, and witness detection for each wrong construction.
- [ ] Run focused tests RED.
- [ ] Implement a bounded transition DSL using immutable tuples/dataclasses only; no `eval`, subprocess, or arbitrary Python execution.
- [ ] Run focused tests GREEN plus behavioral-court tests.
- [ ] Commit `feat(exp298): add bounded code invariant domain`.

### Task 3: Causal-Diagnosis Domain

**Files:**
- Create: `src/nolane_ai/experiments/exp298_causal_worlds.py`
- Test: `tests/test_exp298_causal_worlds.py`

**Interfaces:**
- Produces: `CausalDiagnosisProblem`, `CausalDiagnosisSemantics`, `generate_exp298_causal_world(seed: int)`.

- [ ] Write failing tests for DAG validity, deterministic interventions, 16 balanced candidates, frozen strata, and at least one `observationally_equivalent_interventionally_wrong` candidate per replicate.
- [ ] Add a test showing the observationally equivalent trap matches under no intervention but diverges under an intervention witness.
- [ ] Implement finite deterministic SCM evaluation in topological order and canonical exogenous×intervention probes.
- [ ] Run focused tests GREEN and compile.
- [ ] Commit `feat(exp298): add causal diagnosis domain`.

### Task 4: Grounded-Language Ambiguity Domain

**Files:**
- Create: `src/nolane_ai/experiments/exp298_language_worlds.py`
- Test: `tests/test_exp298_language_worlds.py`

**Interfaces:**
- Produces: `GroundedLanguageProblem`, `GroundedLanguageSemantics`, `generate_exp298_language_world(seed: int)`.

- [ ] Write failing tests for finite micro-world enumeration, deterministic 16-case batches, balanced classes, every frozen ambiguity family across seeds, and exact witnesses for wrong candidates.
- [ ] Add explicit tests for quantifier scope, negation scope, relation direction, referent swap, attachment, conjunction/disjunction, and existential/universal differences.
- [ ] Implement a controlled compositional AST/evaluator; do not call an LLM or use free-form language parsing.
- [ ] Run focused tests GREEN with behavioral-court tests.
- [ ] Commit `feat(exp298): add grounded language ambiguity domain`.

### Task 5: Matched Neural Arms and Pair Audit

**Files:**
- Create: `src/nolane_ai/experiments/matched_cross_domain_fidelity_arms.py`
- Test: `tests/test_matched_cross_domain_fidelity_arms.py`

**Interfaces:**
- Produces: `build_matched_cross_domain_fidelity_arms(...)`, `audit_matched_cross_domain_fidelity_arms(control, fidelity)`, and arm `.decide(...)` methods.

- [ ] Write RED tests requiring equal total/functional/active/optimizer-visible parameter counts, identical initialization digest, equal neural-accounted FLOPs, and a single receipt-access intervention.
- [ ] Implement the smallest EXP-297-pattern-compatible envelope using shared source/candidate encoders, recurrent comparison core, authority/verifier heads, and reserved-parameter finalization to exactly `500000` parameters.
- [ ] Enforce fidelity authority as `executable and receipt.decision == 'court_accept'`; control authority depends only on executability.
- [ ] Run focused tests GREEN.
- [ ] Commit `feat(exp298): add matched cross-domain fidelity arms`.

### Task 6: Deterministic Root Runner and Reconstruction Validator

**Files:**
- Create: `src/nolane_ai/experiments/exp298_paired_runner.py`
- Test: `tests/test_exp298_paired_runner.py`
- Test: `tests/test_exp298_tamper_validation.py`

**Interfaces:**
- Produces: `run_exp298_root(...) -> dict[str, Any]`, `validate_exp298_root(payload) -> list[str]`, schema `NLM-EXP-298-CROSS-DOMAIN-FIDELITY-ROOT-V1`.

- [ ] Write RED runner tests for 3-domain coverage, deterministic lineage, candidate cardinality, raw arm-input leakage receipts, cost fields, domain confusion matrices, BA gains, worst-domain primary, and false evidence flags.
- [ ] Write RED tamper tests that rehash after changing domain id, row order, evaluator label, witness, authority, cost, aggregate, candidate/domain cardinality, model-match receipt, non-claim flag, or successor authorization.
- [ ] Implement root execution using `derive_stream_seed(root_seed, 'EXP-298', ...)`, domain generators, generic court, and matched arms.
- [ ] Implement validator by regenerating every expected source/candidate and recomputing every metric/decision; never trust declared aggregates.
- [ ] Run both focused files GREEN.
- [ ] Commit `feat(exp298): add reconstructable root runner`.

### Task 7: Cross Reducer, Canonical Publication and CLI

**Files:**
- Create: `src/nolane_ai/experiments/exp298_cross_reducer.py`
- Create: `scripts/run_exp298_cross_domain_fidelity_dev.py`
- Test: `tests/test_exp298_cross_reducer.py`
- Test: `tests/test_exp298_paired_cli.py`

**Interfaces:**
- Produces: `reduce_exp298_roots(root_receipts)`, `validate_exp298_cross(payload)`, schema `NLM-EXP-298-CROSS-DOMAIN-FIDELITY-CROSS-V1`.

- [ ] RED tests: reducer accepts exactly roots `[0,1,2,3]`, rejects duplicates/missing roots/identity drift, authorizes EXP-299 design only on 4/4 established roots, and keeps EXP-300 unauthorized.
- [ ] RED CLI tests: frozen defaults only, no tuning flags, no overwrite, atomic canonical JSON, SHA256 sidecar, repository-head/source-tree binding.
- [ ] Implement reducer and CLI. CLI supports root mode `--canonical-index {0,1,2,3}` and cross mode consuming exactly four root receipt paths; scientific geometry is not CLI-configurable.
- [ ] Run focused tests GREEN.
- [ ] Commit `feat(exp298): add cross reducer and hardened cli`.

### Task 8: Frozen Geometry, Workflow Contracts and Freeze Guard

**Files:**
- Create: `protocols/exp298_cross_domain_fidelity_geometry_v1.json`
- Create: `protocols/exp298_cross_domain_fidelity_geometry_v1.sha256`
- Create: `.github/workflows/exp298-cross-domain-fidelity-ci.yml`
- Create: `.github/workflows/exp298-cross-domain-fidelity-development.yml`
- Create: `.github/workflows/exp298-cross-domain-fidelity-freeze-guard.yml`
- Test: `tests/test_exp298_workflow_contract.py`
- Test: `tests/test_exp298_freeze_guard_contract.py`

**Interfaces:** frozen geometry digest and marker path `protocols/exp298-cross-domain-fidelity-release.lock`.

- [ ] RED tests require exact frozen geometry, canonical digest, marker absence before release, exact branch/path triggers, no `workflow_dispatch`, four root jobs + one cross job, canonical artifact names, and freeze guard uniqueness logic.
- [ ] Implement geometry files and workflows following EXP-290's one-shot exact-head discipline but with EXP-298 paths/digests.
- [ ] Dedicated CI must run all EXP-298 tests plus frozen Stage-A verification and compileall. DEVELOPMENT preflight binds marker head, geometry digest, Stage-A digest, repository head, source-tree digest, and first-run uniqueness before model execution.
- [ ] Run contract tests GREEN locally/CI.
- [ ] Commit `ci(exp298): freeze one-shot development court`.

### Task 9: Pre-Data Exact-Head Verification

**Files:** no scientific-code changes unless a genuine pre-data defect is found through RED→GREEN TDD.

- [ ] Run full focused EXP-298 suite and generic CI on exact pre-data head.
- [ ] Verify Stage-A digest remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- [ ] Verify geometry digest and source-tree digest from repository bytes.
- [ ] Verify freeze guard reports zero authoritative DEVELOPMENT runs and marker absent.
- [ ] If any defect is found, add the smallest failing test first, fix only that defect, rerun all exact-head gates, and record the RED/GREEN chain in PR body.

### Task 10: One-Shot DEVELOPMENT Execution and Closure

**Files:**
- Create only after all Task 9 gates GREEN: `protocols/exp298-cross-domain-fidelity-release.lock`
- Later closure docs: `docs/superpowers/specs/2026-09-13-exp298-development-closure.md` and program matrix update on a separate docs-only branch from then-current `main`.

- [ ] Commit the release marker as a marker-only commit containing frozen geometry/protocol/source/pre-marker identities.
- [ ] Verify compare diff is exactly one marker file.
- [ ] Observe exactly one authoritative DEVELOPMENT workflow attempt: four canonical root artifacts then one cross artifact.
- [ ] Run post-release freeze guard and prove no duplicate run.
- [ ] Independently recompute sidecar SHA256, canonical JSON bytes, internal artifact digests, root decisions, cross decision, counts, model-state/evidence flags, and authorization scope from downloaded artifacts.
- [ ] Update experiment PR with sealed metrics/artifact ledger and close it **without merge**.
- [ ] Create a docs-only closure branch from current `main`, copy frozen design/plan/closure record, update the V0.16.1 matrix, run generic CI, review, and merge only the documentation closure.

## Self-review

Coverage check: common court → Tasks 1/6; all three domains → Tasks 2-4; matched intervention → Task 5; frozen metrics/evidence/cost and reconstruction → Task 6; cross authority → Task 7; one-shot discipline → Tasks 8-10. No task modifies frozen Stage-A protocol. No post-result tuning path exists. Function/type names are consistent across task boundaries.