# EXP-297 Encoding Fidelity Court Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EV-E2 / UNVERIFIED matched-neural DEVELOPMENT lane for frozen EXP-297 with diverse compile-valid semantic traps, fail-closed bidirectional witnesses, reconstructible safety metrics, registry/CLI/CI integration, and package 0.15.0.

**Architecture:** Keep the frozen Stage-A protocol byte-identical. Generate deterministic evaluator-labeled candidate courts, expose only label-free CPS/compile/witness information to matched arms, and use the existing `FidelityCourtRegion` inside an equal-geometry neural pair. A reconstruction validator regenerates raw evidence and recomputes all semantic/cost/endpoint claims rather than trusting artifact aggregates.

**Tech Stack:** Python 3.11+, dataclasses, hashlib/json, PyTorch 2.7+ for model-smoke, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-07-exp297-encoding-fidelity-court-design.md`

## Global Constraints

- Do not modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.
- Frozen arms: `compile_only`, `fidelity_court`.
- Primary: `semantic_fidelity_balanced_accuracy`, direction higher.
- Protected: `wrong_formalization_authority_rate <= 0.05`; `faithful_formalization_rejection_rate <= 0.10`.
- MESI: absolute `+0.10` balanced-accuracy fraction; DEVELOPMENT MUST NOT execute that confirmatory decision rule.
- Artifact: `NLM-EXP-297-PAIRED-DEV-EVAL-V1`; evidence `EV-E2`; decision `UNVERIFIED`.
- All confirmatory/challenge/authority-promotion flags remain false.
- `court_inconclusive` never grants authority.
- `hidden_trap_family_consumed=false`; construction labels are evaluator-only.

---

### Task 1: Fail-closed semantic witness court

**Files:**
- Modify: `src/nolane_ai/reasoning/fidelity.py`
- Test: `tests/test_fidelity.py`

**Interfaces:**
- Produces `FidelityDecision`, `FidelityWitness`, `FidelityCourtReceipt`, `canonical_problem_payload()`, `canonical_problem_digest()`, `FidelityCourt.adjudicate(source, candidate)`.
- Preserves `compile_valid()` and `FidelityCourt.accept()` compatibility.

- [ ] Write failing tests asserting faithful equivalence returns `court_accept`, semantic divergence returns `court_reject` with a valid directional witness, and an assignment space above the ceiling returns `court_inconclusive` rather than accept.
- [ ] Run `pytest tests/test_fidelity.py -q`; expected RED because receipt/adjudication interfaces do not exist.
- [ ] Implement exact assignment-space sizing, canonical CPS digest, typed receipt, directional witness search, constraint-check accounting, and fail-closed `accept()` wrapper.
- [ ] Re-run `pytest tests/test_fidelity.py -q`; expected PASS.
- [ ] Commit.

### Task 2: Deterministic multi-stratum candidate court

**Files:**
- Create: `src/nolane_ai/experiments/exp297_fidelity_worlds.py`
- Test: `tests/test_exp297_fidelity_worlds.py`

**Interfaces:**
- Produces `FidelityCandidateCase`, `FidelityWorldBatch`, `generate_fidelity_world(seed)` and canonical arm-view serialization.
- Strata are exactly `faithful_equivalent`, `relation_shift`, `constraint_drop`, `constraint_strengthen`, `constraint_weaken`, `variable_binding_swap`, `domain_mapping_error`, `negation_or_relation_flip`.

- [ ] Write failing tests for deterministic regeneration, all eight strata, compile-valid candidates, faithful/wrong class support, stable candidate order/digests, and evaluator metadata separation from arm views.
- [ ] Run focused test; expected RED because module is absent.
- [ ] Implement table-constraint transformations whose construction provenance determines truth independently from the court.
- [ ] Re-run focused test; expected PASS.
- [ ] Commit.

### Task 3: Matched FidelityCourtRegion arm pair

**Files:**
- Create: `src/nolane_ai/experiments/matched_fidelity_arms.py`
- Test: `tests/test_exp297_neural_import.py`
- Test: `tests/test_matched_fidelity_arms.py`

**Interfaces:**
- Produces `MatchedFidelityArm`, `build_matched_fidelity_arms()`, `audit_matched_fidelity_arms()`, `FidelityArmDecision`.
- Both arms expose identical total/functional/active/optimizer-visible counts and trainable tensor digest.

- [ ] Write RED tests importing the module, verifying identical geometry/initialization, same neural accounting, and canonical null receipt for `compile_only`.
- [ ] Run focused tests; expected RED.
- [ ] Implement deterministic structural CPS features, recurrent comparison layer, existing `FidelityCourtRegion`, receipt adapter, authority/verifier heads, analytical neural accounting, and label-free arm inputs.
- [ ] Ensure deterministic semantic result remains the authority safety gate: fidelity reject/inconclusive cannot be overridden by neural score.
- [ ] Re-run focused tests; expected PASS.
- [ ] Commit.

### Task 4: Paired runner and reconstruction validator

**Files:**
- Create: `src/nolane_ai/experiments/exp297_paired_runner.py`
- Test: `tests/test_exp297_paired_runner.py`
- Test: `tests/test_exp297_ground_truth_boundary.py`

**Interfaces:**
- Produces `run_exp297_paired_development()`, `validate_exp297_execution()`, `build_exp297_execution()` and raw per-candidate rows.

- [ ] Write RED tests for artifact schema/flags, byte-identical candidate lineage, distinct neural vs semantic cost ledgers, fail-closed inconclusive accounting, confusion-matrix reconstruction, BA/protected endpoint reconstruction, and no evaluator label/trap family in arm-observable input.
- [ ] Write RED tamper tests for label, stratum, witness, witness assignment, authority, cost, inconclusive-to-accept, reorder/drop, digest, and forbidden flags.
- [ ] Run focused runner tests; expected RED.
- [ ] Implement deterministic runner and independent reconstruction validator. Validator regenerates each world from seed, replays court semantics, recomputes costs/confusion/endpoints, and validates aggregate digests.
- [ ] Re-run focused tests; expected PASS.
- [ ] Commit.

### Task 5: Registry admission without scientific promotion

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Test: `tests/test_exp297_registry_integration.py`
- Modify: `tests/test_neural_arm_registry.py`

**Interfaces:**
- Adds EXP-297 expected arms and pair audit validation.
- Pair-only status: `PARAMETER_COMPUTE_FIDELITY_COURT_CLOSED`.
- Execution status: `PAIRED_FIDELITY_COURT_DEV_READY`.
- `match_court` remains `BLOCKED`.

- [ ] Add RED registry tests requiring EXP-297, exact frozen arm descriptions, false scientific flags, pair/execution digest binding, and rejection of tampered execution.
- [ ] Run focused registry tests; expected RED.
- [ ] Extend target experiment maps, implementation metadata, pair/execution status computation and validation without weakening EXP-277/279/282/286/289 checks.
- [ ] Re-run focused registry tests; expected PASS.
- [ ] Commit.

### Task 6: Hardened CLI publication

**Files:**
- Create: `scripts/run_exp297_paired_dev.py`
- Test: `tests/test_exp297_paired_cli.py`

**Interfaces:**
- CLI accepts seed/replicate/model-size/verification-ceiling/tiny/output/registry-output controls.
- Verifies canonical Stage-A protocol identity before model work.

- [ ] Write RED tests for successful tiny publication, no-overwrite, frozen protocol digest mismatch, invalid output, and staged all-or-nothing publication.
- [ ] Run CLI tests; expected RED.
- [ ] Implement parser, protocol digest verification, DEVELOPMENT runner call, execution validation, registry validation, temporary-file staging and atomic replace.
- [ ] Re-run CLI tests; expected PASS.
- [ ] Commit.

### Task 7: Documentation, package and CI integration

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

**Interfaces:**
- Package version becomes `0.15.0`.
- Core matrix includes non-model EXP-297 tests; model-smoke includes matched neural/runner/registry/CLI plus tiny runner execution.

- [ ] Update version in both package locations.
- [ ] Add README section stating DEVELOPMENT/non-claim boundary, fail-closed inconclusive semantics, cost separation and smoke command.
- [ ] Add EXP-297 tests and tiny CLI invocation to CI without changing frozen protocol files.
- [ ] Run full core and model-smoke workflows through GitHub Actions; expected GREEN on Python 3.11/3.13 and model-smoke.
- [ ] Commit.

### Task 8: Exact-head review and merge

**Files:** none unless review finds a defect.

- [ ] Verify branch diff contains no changes to frozen protocol bytes.
- [ ] Verify all tamper tests remain present and pass.
- [ ] Review artifact vocabulary for accidental `VERIFIED`, confirmatory/challenge consumption, or semantic-authority promotion.
- [ ] Run complete GitHub Actions at exact head and inspect every required job.
- [ ] Open PR with scientific-boundary summary and test evidence.
- [ ] Squash merge only after required checks are green.
- [ ] Verify post-merge `main` CI at exact squash commit.

## Self-review

Coverage check: all approved design sections map to Tasks 1-8. Placeholder scan: none. Interface consistency: `adjudicate()` feeds generator/runner validation; matched-arm decisions never receive evaluator labels; registry consumes pair/execution artifacts; CLI validates both before publishing. The frozen protocol is read-only throughout.
