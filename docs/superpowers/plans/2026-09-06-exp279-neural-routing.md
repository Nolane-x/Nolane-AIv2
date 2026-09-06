# EXP-279 Neural Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen EXP-279 comparing `propagation_only`, `branch_only`, and `hybrid` under exact parameter matching, shared accounted-FLOP ceiling, paired worlds, and predeclared structure-fit strata.

**Architecture:** Reuse the experiment-local matched-neural patterns proven by EXP-277. Implement one shared tri-arm substrate with computation gating, a deterministic EXP-279 routing generator with three structure-fit strata, a paired runner/semantic validator, registry integration, then a CPU-safe CLI and CI closure.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed utilities.

**Spec:** `docs/superpowers/specs/2026-09-06-exp279-neural-routing-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-279 development artifacts are `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`.
- Frozen primary endpoint is `verified_utility_per_accounted_flop_on_structure_dense_stratum`.
- Frozen MESI is hybrid relative gain `0.08` against the best simpler arm.
- Frozen protected floor is `hybrid >= best_simple - 0.01` on verified solution rate.
- Resource match requires reclaimed parameters assigned to simpler rivals, equal max accounted FLOPs, and predeclared structure-fit strata.
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

- [ ] **Step 1: Write the failing import test**

```python
import pytest
pytest.importorskip("torch")


def test_exp279_neural_modules_import() -> None:
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development
    from nolane_ai.experiments.matched_routing_arms import build_matched_exp279_arm_triplet

    assert callable(run_exp279_paired_development)
    assert callable(build_matched_exp279_arm_triplet)
```

- [ ] **Step 2: Add the test to model-smoke**

Append `tests/test_exp279_neural_import.py` to the explicit model-smoke pytest list.

- [ ] **Step 3: Push and preserve RED**

Expected GitHub Actions outcome: core 3.11/3.13 green, model-smoke fails only with missing EXP-279 neural module. Do not implement production files before this RED is recorded.

---

### Task 2: Matched tri-arm neural substrate and resource ledger

**Files:**
- Create: `src/nolane_ai/experiments/matched_routing_arms.py`
- Create: `tests/test_matched_routing_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `PropagationOnlyArm`, `BranchOnlyArm`, `HybridRoutingArm`.
- Produces `build_matched_exp279_arm_triplet(...)` and `audit_matched_exp279_arm_triplet(...)`.
- Output dataclass fields: `decision_logits`, `verifier_confidence`, `residual_uncertainty`, `branch_route_mask`, `representation_semantics`.

- [ ] **Step 1: Write RED behavior tests**

Tests must assert:

```python
prop, branch, hybrid = build_matched_exp279_arm_triplet(
    d_model=8,
    hidden_size=8,
    target_parameters=4096,
)
audit = audit_matched_exp279_arm_triplet(
    prop,
    branch,
    hybrid,
    timesteps=3,
    variables=4,
    constraints=2,
    max_accounted_flops_per_episode=None,
)
assert audit["parameter_match"] is True
assert audit["functional_parameter_match"] is True
assert audit["compute_budget_closed"] is True
assert audit["compute_ledger"]["propagation_only"]["hardware_profiler_flops_claimed"] is False
assert audit["compute_ledger"]["branch_only"]["hardware_profiler_flops_claimed"] is False
assert audit["compute_ledger"]["hybrid"]["hardware_profiler_flops_claimed"] is False
```

Also assert `branch_only` rejects an `incidence=` keyword, while propagation/hybrid require rank-3 incidence. All three arms must return identical decision-logit shape.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_matched_routing_arms.py` and confirm missing module/classes.

- [ ] **Step 3: Implement one shared parameter envelope**

Create a `_MatchedExp279ArmBase` containing the complete functional module inventory. Use `finalize_region_budget(..., frozen=False)` for exact target closure. `build_matched_exp279_arm_triplet` constructs the three arms under one seeded architecture and clones identical state dictionaries.

- [ ] **Step 4: Implement arm execution semantics**

- propagation-only: incidence message passing, no branch GRU execution;
- branch-only: recurrent branch refinement from surface/variable representation, no incidence argument;
- hybrid: propagation first, residual uncertainty from propagation logits, branch refinement only for batch episodes above `route_threshold`.

The hybrid output must expose a boolean per-episode route mask.

- [ ] **Step 5: Implement analytical compute/resource audit**

Audit exact total/functional/reclaimed-idle parameter counts and conservative scalar FLOPs for the supplied geometry. Record one shared declared max ceiling and fail if any arm exceeds it. Reclaimed/idle parameter capacity is part of the matched envelope but not executed FLOPs.

- [ ] **Step 6: Verify GREEN**

Run:

`pytest -q tests/test_matched_routing_arms.py tests/test_matched_cbrf_arms.py tests/test_matched_belief_arms.py`

Expected: all pass.

---

### Task 3: Deterministic structure-fit routing generator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_routing_worlds.py`
- Create: `tests/test_exp279_routing_worlds.py`

**Interfaces:**
- Produces `Exp279RoutingBatch` and `Exp279RoutingGenerator.make_batch(...)`.
- Strata enum values are exactly `PROPAGATION_FIT`, `BRANCH_FIT`, `MIXED_RESIDUAL`.
- Batch fields include `surface_events`, `variable_states`, `incidence`, `targets`, `stratum`, `metadata`, `digest`, `replicate`, `rng_stream`.

- [ ] **Step 1: Write determinism/strata RED tests**

For identical root seed, replicate, stream, stratum and geometry, tensors and digest must be byte-identical. Changing replicate, stream, stratum or geometry must change digest. Incidence is rank-3 `[batch,constraints,variables]`, targets are binary, and every supported stratum can be generated.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp279_routing_worlds.py`.

- [ ] **Step 3: Implement generator**

Use only:

```python
seed = derive_stream_seed(root_seed, "EXP-279", replicate, rng_stream)
```

Generate equality/component worlds with deterministic geometry differences by stratum. `PROPAGATION_FIT` exposes strongly aligned incidence/anchor structure; `BRANCH_FIT` weakens direct propagation support while preserving informative surface sequences; `MIXED_RESIDUAL` mixes both component types within the same batch. Hash lineage metadata plus raw tensor bytes using existing tensor-byte utilities.

- [ ] **Step 4: Verify regeneration twice**

Run generator tests twice and require both runs green.

---

### Task 4: Paired development runner and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_paired_runner.py`
- Create: `tests/test_exp279_paired_runner.py`

**Interfaces:**
- Produces `run_exp279_paired_development(...) -> dict[str, Any]`.
- Produces `validate_exp279_paired_development(payload) -> list[str]`.
- Artifact schema is `NLM-EXP-279-PAIRED-DEV-EVAL-V1`.

- [ ] **Step 1: Write RED runner/validator tests**

Assert artifact contains:

```python
assert payload["schema"] == "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
assert payload["evidence_level"] == "EV-E2"
assert payload["decision"] == "UNVERIFIED"
assert payload["confirmatory_ready"] is False
assert payload["confirmatory_data_consumed"] is False
assert payload["challenge_materialized"] is False
assert payload["decision_rule_executed"] is False
assert payload["primary_endpoint"]["metric"] == "verified_utility_per_accounted_flop_on_structure_dense_stratum"
assert payload["primary_endpoint"]["mesi_relative_gain"] == 0.08
assert payload["protected_endpoints"]["verified_solution_rate_floor"] == "hybrid >= best_simple - 0.01"
```

Also require identical initial functional digests, training stream `augmentation`, evaluation stream `evaluation`, disjoint replicate ranges, blocked deterministic strata, unique paired batch digests, resource-match closure, route receipt, and valid artifact self-hash.

- [ ] **Step 2: Add semantic tamper tests**

Deep-copy a valid payload, alter one field, recompute the top-level artifact digest, and require validation failure for each of:

- primary metric/MESI;
- protected floor;
- arm ordering;
- branch-only incidence receipt;
- stratum ordering/membership;
- route threshold;
- route receipt counts/fractions;
- compute ceiling/resource flags;
- aggregate values;
- confirmatory/challenge boundary.

- [ ] **Step 3: Verify RED**

Run `pytest -q tests/test_exp279_paired_runner.py`.

- [ ] **Step 4: Implement paired training**

Seed triplet initialization from `derive_stream_seed(root_seed, "EXP-279", 0, "model_init")`. Train all three arms on exactly the same ordered balanced augmentation batches and identical optimizer hyperparameters.

- [ ] **Step 5: Implement blocked paired evaluation**

Evaluation repeats the stratum cycle `PROPAGATION_FIT`, `BRANCH_FIT`, `MIXED_RESIDUAL` deterministically from `eval_start_replicate`. For each replicate run all arms on one paired batch, externally verify predictions, charge arm-specific accounted FLOPs, record hybrid route mask/fraction, and emit raw per-replicate rows.

- [ ] **Step 6: Implement descriptive aggregates only**

Compute mean utility/solution rate by arm and stratum, best simpler arm descriptively, hybrid relative utility gain and hybrid-minus-best-simple solution-rate difference. Do **not** run bootstrap/Holm inference and do not emit a scientific promotion state.

- [ ] **Step 7: Verify GREEN**

Run:

`pytest -q tests/test_exp279_paired_runner.py tests/test_matched_routing_arms.py tests/test_exp279_routing_worlds.py`

---

### Task 5: Neural arm registry integration

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Create: `tests/test_exp279_registry_integration.py`
- Modify: `tests/test_neural_arm_registry.py`

**Interfaces:**
- Extend `build_neural_arm_registry(..., exp279_pair_audit=None, exp279_execution_artifact=None, ...)`.

- [ ] **Step 1: Write RED registry tests**

Valid pair audit clears implementation blockers for all three EXP-279 arms and records parameter/reclaimed-capacity/compute evidence. Valid execution artifact advances `development_match_status` to `PAIRED_ROUTING_DEV_READY`. Supplying execution without audit, invalid execution, protocol mismatch or failed compute closure must raise `ValueError`.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp279_registry_integration.py tests/test_neural_arm_registry.py`.

- [ ] **Step 3: Implement registry wiring**

Add `_validate_exp279_pair_audit`. On valid development evidence set each EXP-279 arm to `IMPLEMENTED` / `MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM`, but keep `match_court = BLOCKED`.

With execution evidence, blockers become exactly development-external scientific debts: confirmatory sample-size/blocked-analysis freeze, confirmatory-open execution, and post-freeze challenge evidence.

- [ ] **Step 4: Verify GREEN**

Run all neural registry tests plus all EXP-277/EXP-279/EXP-282 paired-runner tests.

---

### Task 6: CLI, version, CI closure, review and merge

**Files:**
- Create: `scripts/run_exp279_paired_dev.py`
- Create: `tests/test_exp279_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`
- Modify: `README.md`

**Interfaces:**
- CLI writes one append-only EXP-279 development artifact plus optional registry output.

- [ ] **Step 1: Write CLI RED tests**

Test tiny execution, canonical protocol verification, output overwrite refusal, EV-E2 boundary, and absence of confirmatory/challenge consumption.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp279_paired_cli.py`.

- [ ] **Step 3: Implement CLI**

Before model creation:

1. resolve repository root;
2. verify canonical `protocols/stage_a_v1.json` against its `.sha256` authority;
3. refuse existing output paths;
4. compute source-tree digest;
5. execute tiny/default development geometry;
6. validate artifact before writing;
7. optionally build and write registry snapshot.

- [ ] **Step 4: Wire CI**

Add all EXP-279 test files to the explicit model-smoke pytest command and one CPU-safe command:

```bash
python scripts/run_exp279_paired_dev.py \
  --tiny \
  --train-replicates 3 \
  --eval-replicates 3 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 4 \
  --constraints 2 \
  --output /tmp/nlm-exp279-paired.json \
  --registry-output /tmp/nlm-exp279-neural-arm-registry.json
```

- [ ] **Step 5: Version only after EXP-279 lane is green**

Change package version from `0.11.0` to `0.12.0` in both version authorities used by the repo.

- [ ] **Step 6: Update README scientific boundary**

Document the development command, three-arm information separation, predeclared strata and `EV-E2 / UNVERIFIED` boundary. Do not describe a development outcome as scientific evidence.

- [ ] **Step 7: Full exact-head verification**

Require GitHub Actions core 3.11, core 3.13, and model-smoke all green. Confirm all EXP-277 and EXP-282 ceremony regressions remain green.

- [ ] **Step 8: Diff guard**

Use the PR changed-file list/compare and confirm neither `protocols/stage_a_v1.json` nor `protocols/stage_a_v1.sha256` appears.

- [ ] **Step 9: Reviewer pass**

Review information separation, route semantics, stratum predeclaration, aggregate recomputation, semantic tamper resistance, resource accounting and scientific-language boundaries. Fix every Critical/Important issue before merge.

- [ ] **Step 10: Squash merge exact head**

Merge only with `expected_head_sha` equal to the verified green PR head. Then check post-merge `main` CI before claiming closure.
