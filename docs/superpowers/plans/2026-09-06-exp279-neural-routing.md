# EXP-279 Neural Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen EXP-279 comparing `propagation_only`, `branch_only`, and `hybrid` under exact total/functional/active-functional parameter matching, shared accounted-FLOP ceiling, paired worlds, and predeclared structure-fit strata.

**Architecture:** Reuse the experiment-local matched-neural patterns proven by EXP-277. Implement one shared tri-arm substrate in which every non-reserve functional parameter has an active semantic assignment in every arm, a deterministic EXP-279 routing generator with three structure-fit strata, a paired runner/semantic validator, registry integration, then a CPU-safe CLI and CI closure.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed utilities.

**Spec:** `docs/superpowers/specs/2026-09-06-exp279-neural-routing-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-279 development artifacts are `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`.
- Frozen primary endpoint is `verified_utility_per_accounted_flop_on_structure_dense_stratum`.
- Frozen MESI is hybrid relative gain `0.08` against the best simpler arm.
- Frozen protected floor is `hybrid >= best_simple - 0.01` on verified solution rate.
- Resource match requires reclaimed parameters assigned to simpler rivals, equal max accounted FLOPs, and predeclared structure-fit strata.
- Every non-reserve functional parameter must have an active semantic assignment in every arm; `capacity_reserve` alone may remain non-functional.
- Training RNG stream is `augmentation`; development evaluation RNG stream is `evaluation`.
- No confirmatory-open observations or post-freeze challenge randomness may be materialized.
- Hardware-profiler FLOPs are not claimed; accounting is analytical scalar arithmetic.

---

### Task 1: Establish clean RED import contract

**Files:**
- Create: `tests/test_exp279_neural_import.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces the CI-enforced expectation that `nolane_ai.experiments.matched_routing_arms` and `nolane_ai.experiments.exp279_paired_runner` exist.

- [x] **Step 1: Write the failing import test**

- [x] **Step 2: Add the test to model-smoke**

- [x] **Step 3: Preserve clean RED**

GitHub Actions run #136: core 3.11/3.13 green; model-smoke failed only because `nolane_ai.experiments.exp279_paired_runner` did not exist. Result: `1 failed, 143 passed`.

---

### Task 2: Matched tri-arm neural substrate and resource ledger

**Files:**
- Create: `src/nolane_ai/experiments/matched_routing_arms.py`
- Test: `tests/test_matched_routing_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `PropagationOnlyArm`, `BranchOnlyArm`, `HybridRoutingArm`.
- Produces `build_matched_exp279_arm_triplet(...)` and `audit_matched_exp279_arm_triplet(...)`.
- Output dataclass fields: `decision_logits`, `verifier_confidence`, `residual_uncertainty`, `branch_route_mask`, `representation_semantics`.

- [x] **Step 1: Write RED behavior tests**

Tests require exact total/functional/active-functional equality, explicit reclaimed-parameter assignment, shared compute ceiling, no hardware-profiler FLOP claim, exact output shapes, incidence separation, route-threshold behavior and fail-closed compute budgeting.

- [ ] **Step 2: Verify broad RED**

Run the full explicit model-smoke test list before adding production modules. Expected failures are missing EXP-279 modules/script only.

- [ ] **Step 3: Implement one shared active parameter envelope**

Create `_MatchedExp279ArmBase` with common surface/variable projections, recurrent core, propagation transform, route scorer, decision/verifier heads and `capacity_reserve` closure via `finalize_region_budget(..., frozen=False)`. Construct all three arms with identical state dictionaries.

- [ ] **Step 4: Implement reclaimed arm semantics**

- `propagation_only`: consumes incidence; uses the propagation transform for constraint-variable message passing, reclaims the recurrent core as iterative propagation refinement, and uses the route scorer as a propagation-confidence gate. It never represents that recurrent refinement as branch search.
- `branch_only`: has no incidence argument; reclaims the propagation transform as a branch preconditioner, uses the recurrent core for branch refinement, and uses the route scorer as a branch-confidence gate.
- `hybrid`: consumes incidence, propagates first, uses the route scorer for residual uncertainty, and conditionally executes recurrent branch refinement only above the frozen `route_threshold`.

The hybrid output exposes a boolean per-episode route mask. Threshold `0.0` routes all episodes; threshold `1.0` routes none.

- [ ] **Step 5: Implement analytical compute/resource audit**

Audit exact total/functional/active-functional counts and per-arm reclaimed semantic assignments. Record conservative scalar FLOPs for each arm, one shared declared ceiling, and a hybrid fully-routed max plus actual-route accounting components. Fail closed if any max exceeds the declared ceiling.

- [ ] **Step 6: Verify GREEN**

Run `tests/test_matched_routing_arms.py` plus existing EXP-277/EXP-282 matched-arm tests.

---

### Task 3: Deterministic structure-fit routing generator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_routing_worlds.py`
- Test: `tests/test_exp279_routing_worlds.py`

**Interfaces:**
- Produces `Exp279RoutingBatch` and `Exp279RoutingGenerator.make_batch(...)`.
- Strata are exactly `PROPAGATION_FIT`, `BRANCH_FIT`, `MIXED_RESIDUAL`.
- Batch fields: `surface_events`, `variable_states`, `incidence`, `targets`, `stratum`, `metadata`, `digest`, `replicate`, `rng_stream`.

- [x] **Step 1: Write determinism/strata RED tests**

- [ ] **Step 2: Verify broad RED**

Expected: missing `exp279_routing_worlds` module before implementation.

- [ ] **Step 3: Implement generator**

Derive randomness only with `derive_stream_seed(root_seed, "EXP-279", replicate, rng_stream)`. Generate truthful component-variable incidence in every stratum while varying structure/surface geometry, never oracle labels. Hash lineage metadata plus raw tensor bytes with existing tensor-byte utilities.

- [ ] **Step 4: Verify regeneration twice**

Run generator tests twice and require both runs green.

---

### Task 4: Paired development runner and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_paired_runner.py`
- Test: `tests/test_exp279_paired_runner.py`

**Interfaces:**
- Produces `run_exp279_paired_development(...) -> dict[str, Any]`.
- Produces `validate_exp279_paired_development(payload) -> list[str]`.
- Artifact schema: `NLM-EXP-279-PAIRED-DEV-EVAL-V1`.

- [x] **Step 1: Write runner/validator RED tests**
- [x] **Step 2: Write semantic tamper tests**
- [ ] **Step 3: Verify broad RED**

- [ ] **Step 4: Implement paired training**

Seed identical triplet state from `derive_stream_seed(root_seed, "EXP-279", 0, "model_init")`. Train all arms on the same balanced augmentation batches and identical optimizer hyperparameters.

- [ ] **Step 5: Implement blocked paired evaluation**

Cycle evaluation strata deterministically `PROPAGATION_FIT -> BRANCH_FIT -> MIXED_RESIDUAL`. Run all arms on one paired batch, externally verify targets, charge arm-specific FLOPs, and record hybrid route mask/count/fraction.

- [ ] **Step 6: Implement descriptive aggregates only**

Compute raw/mean arm metrics, deterministic best-simple arm, hybrid relative utility gain, hybrid-minus-best-simple solution-rate difference and by-stratum summaries. Do not run bootstrap/Holm confirmatory inference and do not emit promotion/kill decisions.

- [ ] **Step 7: Verify GREEN**

Run runner + matched arms + generator tests together.

---

### Task 5: Neural arm registry integration

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Test: `tests/test_exp279_registry_integration.py`
- Modify: `tests/test_neural_arm_registry.py` only if shared validator coverage requires it.

**Interfaces:**
- Extend `build_neural_arm_registry(..., exp279_pair_audit=None, exp279_execution_artifact=None, ...)`.

- [x] **Step 1: Write RED registry tests**
- [ ] **Step 2: Verify broad RED**
- [ ] **Step 3: Implement registry wiring**

Validate pair-audit active resource closure and execution semantics. Valid pair evidence sets all three EXP-279 arms to `IMPLEMENTED / MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM` with `PARAMETER_RECLAIM_COMPUTE_STRATA_CLOSED`; valid execution advances development status to `PAIRED_ROUTING_DEV_READY`. `match_court` remains `BLOCKED`.

- [ ] **Step 4: Verify GREEN**

Run neural registry tests plus EXP-277/279/282 paired-runner tests.

---

### Task 6: CLI, version, CI closure, review and merge

**Files:**
- Create: `scripts/run_exp279_paired_dev.py`
- Test: `tests/test_exp279_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`
- Modify: `README.md`

**Interfaces:**
- CLI writes one append-only EXP-279 development artifact and one registry snapshot.

- [x] **Step 1: Write CLI RED tests**
- [ ] **Step 2: Verify broad RED**
- [ ] **Step 3: Implement CLI**

Refuse existing outputs before protocol/model execution; verify canonical frozen Stage-A digest; compute source-tree digest; run tiny/default development geometry; semantically validate before writing; build exact-16M registry audit on meta; never derive confirmatory/challenge seeds.

- [ ] **Step 4: Wire CI**

Add all EXP-279 tests plus one CPU-safe `run_exp279_paired_dev.py --tiny --train-replicates 3 --eval-replicates 3 ...` smoke.

- [ ] **Step 5: Version only after EXP-279 lane is green**

Bump both package authorities from `0.11.0` to `0.12.0`.

- [ ] **Step 6: Update README scientific boundary**

Document development use, three-arm information separation, active parameter reclaim, predeclared strata, and EV-E2/UNVERIFIED limits.

- [ ] **Step 7: Full exact-head verification**

Require core 3.11, core 3.13 and model-smoke all green, including EXP-277 and EXP-282 regressions.

- [ ] **Step 8: Diff guard**

Confirm neither frozen Stage-A protocol file appears in the PR diff.

- [ ] **Step 9: Reviewer pass**

Review information separation, active parameter assignment, route semantics, predeclared strata, aggregate recomputation, tamper resistance, compute accounting and scientific-language boundaries. Fix every Critical/Important issue.

- [ ] **Step 10: Squash merge exact head and verify post-merge main CI**

Merge with `expected_head_sha` equal to the fully-green head, then verify post-merge `main` CI before claiming closure.
