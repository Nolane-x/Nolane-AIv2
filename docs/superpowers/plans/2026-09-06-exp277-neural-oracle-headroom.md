# EXP-277 Neural Oracle-Structure Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen EXP-277 that compares a matched ARCS recurrent branch arm against an oracle-structure CBRF arm without touching frozen protocol bytes or consuming confirmatory/challenge data.

**Architecture:** Add experiment-local matched arms, a deterministic paired structure-dense generator, a self-validating paired-development runner, then wire validated evidence into the existing neural-arm registry. Reuse the EXP-282 lineage/digest/resource-audit patterns, but keep EXP-277 semantics and metrics distinct.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed utilities.

**Spec:** `docs/superpowers/specs/2026-09-06-exp277-neural-oracle-headroom-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-277 development artifacts are `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`.
- Frozen MESI = 0.10 relative gain on `verified_utility_per_accounted_flop`.
- Protected solution-rate floor = `oracle_cbrf >= arcs_branch - 0.005`.
- Training RNG stream = `augmentation`; development evaluation RNG stream = `evaluation`.
- No confirmatory-open or post-freeze-challenge randomness is materialized.
- Exact trainable parameter equality is mandatory; all executed compute is ledgered honestly.

---

### Task 1: Establish RED import contract

**Files:**
- Create: `tests/test_exp277_neural_import.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: frozen EXP-277 arm IDs from Stage-A V1.
- Produces: CI-enforced expectation that `nolane_ai.experiments.matched_cbrf_arms` and `nolane_ai.experiments.exp277_paired_runner` exist.

- [ ] **Step 1: Write failing import test**

```python
import pytest
pytest.importorskip("torch")


def test_exp277_neural_modules_import():
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development
    from nolane_ai.experiments.matched_cbrf_arms import build_matched_exp277_arm_pair
    assert callable(run_exp277_paired_development)
    assert callable(build_matched_exp277_arm_pair)
```

- [ ] **Step 2: Add the test to the explicit `model-smoke` pytest command**

Append `tests/test_exp277_neural_import.py` to the model-smoke test list.

- [ ] **Step 3: Run CI**

Expected: core jobs green; model-smoke fails with missing EXP-277 neural module. Preserve this RED run as TDD evidence.

---

### Task 2: Matched EXP-277 neural arms and compute ledger

**Files:**
- Create: `src/nolane_ai/experiments/matched_cbrf_arms.py`
- Create: `tests/test_matched_cbrf_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `ARCSBranchArm`, `OracleCBRFArm`, `build_matched_exp277_arm_pair(...)`, `audit_matched_exp277_arm_pair(...)`.
- Both forward paths return per-variable binary logits and verifier confidence; only oracle forward accepts compiled incidence.

- [ ] **Step 1: Write tests**

Tests assert exact total/functional parameter equality, identical output shape, ARCS rejects any `incidence=` argument, oracle requires rank-3 incidence, pair audit reports `parameter_match=true`, and both ledgers declare `hardware_profiler_flops_claimed=false`.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_matched_cbrf_arms.py`; expected missing module/classes.

- [ ] **Step 3: Implement minimal matched arms**

Use one shared experiment-local base with exact target-parameter reserve closure. `ARCSBranchArm` uses recurrent deliberation over surface events and variable query states. `OracleCBRFArm` uses the same input/output width and target parameter count but adds factor-incidence message passing before the decision/verifier heads. Keep reserve tensors out of optimizer-visible functional parameters via existing `functional_trainable_named_parameters` conventions.

- [ ] **Step 4: Implement analytical ledgers**

Each audit reports exact parameter counts and conservative scalar arithmetic counts for the supplied geometry. The pair audit records one shared declared max-FLOP budget and fails if either execution exceeds it.

- [ ] **Step 5: Verify GREEN**

Run `pytest -q tests/test_matched_cbrf_arms.py` and existing EXP-282 matched-arm tests.

---

### Task 3: Deterministic structure-dense neural generator

**Files:**
- Create: `src/nolane_ai/experiments/exp277_structure_dense.py`
- Create: `tests/test_exp277_structure_dense.py`

**Interfaces:**
- Produces: `Exp277StructureDenseBatch` and `Exp277StructureDenseGenerator.make_batch(...)`.
- Batch fields: `surface_events`, `variable_states`, `oracle_incidence`, `targets`, `digest`, `replicate`, `rng_stream`.

- [ ] **Step 1: Write determinism/lineage tests**

Same root seed + experiment + replicate + stream + geometry must regenerate byte-identical tensors/digest; changing replicate, stream or geometry must change digest. Batch targets are binary and incidence is rank-3 `[batch,constraints,variables]`.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp277_structure_dense.py`.

- [ ] **Step 3: Implement generator**

Derive randomness only with `derive_stream_seed(root_seed, "EXP-277", replicate, rng_stream)`. Generate multiple equality-connected components with independent anchors; serialize the same facts into a surface/event tensor for ARCS and a compiled incidence tensor for oracle. Hash tensor bytes + geometry + lineage into one deterministic digest.

- [ ] **Step 4: Verify GREEN**

Run generator tests twice to prove regeneration stability.

---

### Task 4: Paired development runner and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp277_paired_runner.py`
- Create: `tests/test_exp277_paired_runner.py`

**Interfaces:**
- Produces: `run_exp277_paired_development(...) -> dict[str, Any]` and `validate_exp277_paired_development(payload) -> list[str]`.
- Consumes matched-arm pair and generator from Tasks 2–3.

- [ ] **Step 1: Write validator/runner tests**

Assert schema `NLM-EXP-277-PAIRED-DEV-EVAL-V1`, `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`, exact parameter match, oracle-information receipt, `augmentation` training stream, `evaluation` eval stream, disjoint replicate ranges, paired unique batch digests, frozen primary/MESI/protected metadata, and self-hash validity.

- [ ] **Step 2: Add tamper tests**

Tamper evaluation ordering, primary metric, MESI, oracle-information receipt, or resource-match fields; recompute top-level digest; semantic validation must still reject.

- [ ] **Step 3: Verify RED**

Run `pytest -q tests/test_exp277_paired_runner.py`.

- [ ] **Step 4: Implement runner**

Seed one matched arm pair from the `model_init` stream, clone functional initialization, train both on exactly the same `augmentation` batches, evaluate both on exactly the same disjoint `evaluation` batches, externally verify per-variable decisions, record each arm's accounted FLOPs, compute per-replicate verified utility/FLOP and oracle-relative gain, and emit no scientific promotion decision.

- [ ] **Step 5: Verify GREEN**

Run paired-runner, matched-arm and generator tests together.

---

### Task 5: Neural arm registry integration

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Modify: `tests/test_neural_arm_registry.py`

**Interfaces:**
- Extend `build_neural_arm_registry(..., exp277_pair_audit=None, exp277_execution_artifact=None, ...)`.

- [ ] **Step 1: Write failing registry tests**

Valid EXP-277 pair audit clears the two arm implementation blockers and records matched-resource evidence. Valid paired execution adds development evidence. Missing audit with supplied execution, invalid artifact, protocol mismatch or compute-budget failure raises.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_neural_arm_registry.py`.

- [ ] **Step 3: Implement minimal registry wiring**

Mirror the EXP-282 evidence-gated integration pattern but keep EXP-277 blocker text and endpoint fields. Even with development evidence, `match_court` remains blocked from confirmatory promotion and lists confirmatory sample-size/analysis/challenge blockers.

- [ ] **Step 4: Verify GREEN**

Run registry tests plus all EXP-277 tests.

---

### Task 6: CLI, package version and CI closure

**Files:**
- Create: `scripts/run_exp277_paired_dev.py`
- Create: `tests/test_exp277_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

**Interfaces:**
- CLI writes one append-only development JSON artifact and refuses overwrite.

- [ ] **Step 1: Write CLI RED tests**

Test tiny execution, frozen protocol verification, output refusal, EV-E2 boundary and no confirmatory/challenge fields indicating consumption/materialization.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp277_paired_cli.py`.

- [ ] **Step 3: Implement CLI**

Verify canonical Stage-A protocol digest before model execution, verify output nonexistence before training, compute source-tree digest, run tiny/default development geometry and persist one artifact.

- [ ] **Step 4: Wire model-smoke**

Add EXP-277 unit/CLI tests and one tiny EXP-277 development command. Do not add confirmatory execution.

- [ ] **Step 5: Version**

Bump package from `0.10.0` to `0.11.0` only after the full EXP-277 development lane is green.

- [ ] **Step 6: Full verification**

Run the exact PR CI matrix. Confirm frozen protocol files are absent from the PR diff, no challenge material is generated, and all existing EXP-282 ceremony tests remain green.

- [ ] **Step 7: Reviewer and squash merge**

Review semantic validation, lineage, resource accounting and scientific-language boundaries. Squash merge only the exact fully-green head.
