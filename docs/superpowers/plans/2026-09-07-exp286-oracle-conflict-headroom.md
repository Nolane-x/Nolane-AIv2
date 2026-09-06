# EXP-286 Oracle Conflict-Core Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen `EXP-286` comparing `chronological_failure` against contradiction-time `oracle_conflict_core` under exact parameter matching, one shared accounted-FLOP ceiling, deterministic paired solvable conflict worlds, censored scientific failures, semantic validation, registry integration, CLI/CI closure, and package `0.13.0`.

**Architecture:** Follow the experiment-local matched-neural pattern already proven by EXP-277/EXP-279. Use two identical neural arm envelopes and identical initial world/model lineage; the only privileged information difference is a ground-truth minimal/local conflict-core receipt delivered to `oracle_conflict_core` after its current contradiction. Preserve causal post-action search divergence, charge all executed neural work analytically, censor unresolved episodes at the shared ceiling, and keep all outputs `EV-E2 / UNVERIFIED`.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed/tensor-byte/optimizer utilities, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-07-exp286-oracle-conflict-headroom-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-286 artifacts remain `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`, `confirmatory_data_consumed=false`, `challenge_materialized=false`, `decision_rule_executed=false`.
- Frozen arms are exactly `chronological_failure` and `oracle_conflict_core`.
- Frozen primary endpoint is `accounted_reasoning_flops_to_verified_solution`, direction `lower`.
- Frozen MESI is a relative FLOP reduction of `0.15`.
- Frozen protected floor is `oracle_conflict_core >= chronological_failure - 0.005` on `verified_solution_rate`.
- Frozen analysis is paired log-cost ratio plus bootstrap CI with failures retained as censored/scientific outcomes; DEVELOPMENT may emit descriptive statistics only.
- Oracle core information may be delivered only after the current contradiction and only to `oracle_conflict_core`.
- Training RNG stream is `augmentation`; DEVELOPMENT evaluation stream is `evaluation`; initialization stream is `model_init`.
- Same replicate means same initial world and exogenous lineage. Search-path divergence caused by arm actions is retained, not normalized away.
- Same maximum accounted FLOP ceiling applies to both arms. Hardware-profiler FLOPs are never claimed.
- Unresolved episodes remain in raw rows with cost set to the common censoring ceiling and `censored_at_max_flops=true`.
- A positive DEVELOPMENT result does not validate learned `ConflictCoreRegion`; a negative result may justify simplifying/removing later conflict machinery.

---

### Task 1: Establish a clean CI-visible RED contract

**Files:**
- Create: `tests/test_exp286_neural_import.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces CI-enforced import expectations for `build_matched_exp286_arm_pair` and `run_exp286_paired_development` before production modules exist.

- [ ] **Step 1: Write the failing import test**

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_exp286_neural_modules_import() -> None:
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development
    from nolane_ai.experiments.matched_conflict_arms import build_matched_exp286_arm_pair

    assert callable(run_exp286_paired_development)
    assert callable(build_matched_exp286_arm_pair)
```

- [ ] **Step 2: Add only this new test to model-smoke**

Append `tests/test_exp286_neural_import.py` to the explicit pytest command. Do not add production files yet.

- [ ] **Step 3: Commit and open the PR**

Commit the RED test plus CI edit, open a PR to `main`, and preserve exact branch head.

- [ ] **Step 4: Verify intentional RED in GitHub Actions**

Expected: core 3.11 and 3.13 stay green; model-smoke fails at the EXP-286 import with `ModuleNotFoundError` for a missing new production module. Existing tests must not fail.

---

### Task 2: Matched neural conflict arms and analytical resource court

**Files:**
- Create: `src/nolane_ai/experiments/matched_conflict_arms.py`
- Create: `tests/test_matched_conflict_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `ChronologicalFailureArm(nn.Module)`.
- Produces `OracleConflictCoreArm(nn.Module)`.
- Produces `Exp286ArmOutput` with fields `rollback_logits`, `verifier_confidence`, `conflict_conditioning_used`, `representation_semantics`.
- Produces `build_matched_exp286_arm_pair(*, d_model: int, hidden_size: int, target_parameters: int, device=None) -> tuple[ChronologicalFailureArm, OracleConflictCoreArm]`.
- Produces `audit_matched_exp286_arm_pair(..., timesteps: int, variables: int, max_search_steps: int, max_accounted_flops_per_episode: int | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write RED behavior/resource tests**

Require:

```python
chronological, oracle = build_matched_exp286_arm_pair(
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
)
audit = audit_matched_exp286_arm_pair(
    chronological,
    oracle,
    timesteps=3,
    variables=5,
    max_search_steps=8,
)
assert audit["schema"] == "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1"
assert audit["parameter_match"] is True
assert audit["functional_parameter_match"] is True
assert audit["active_functional_parameter_match"] is True
assert audit["optimizer_visible_parameter_match"] is True
assert audit["oracle_information_separation"] is True
assert audit["compute_budget_closed"] is True
assert audit["chronological_failure"]["reserved_parameters"] == audit["oracle_conflict_core"]["reserved_parameters"]
assert audit["chronological_failure"]["optimizer_visible_parameters"] == audit["oracle_conflict_core"]["optimizer_visible_parameters"]
```

Also test identical state dictionaries after construction, both arms requiring rank-3 surface/variable inputs, chronological rejecting a real oracle-core mask, oracle rejecting a core mask before `contradiction_observed=True`, and both producing matching rollback-logit shapes.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_matched_conflict_arms.py` and require missing module/symbol failure.

- [ ] **Step 3: Implement one shared trainable envelope**

Create `_MatchedExp286ArmBase` with identical modules in both arms:

```python
self.event_projection = nn.Linear(d_model, hidden_size)
self.variable_projection = nn.Linear(d_model, hidden_size)
self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
self.core_adapter = nn.Linear(1, hidden_size)
self.rollback_head = nn.Linear(hidden_size, 1)
self.verifier_head = nn.Linear(hidden_size, 1)
self.mix_gate = nn.Parameter(torch.zeros(()))
finalize_region_budget(self, target_parameters, device=device, frozen=False)
```

Use `functional_trainable_named_parameters` for optimizer-visible accounting. If `capacity_reserve` exists because target closure requires it, it must be excluded from the claimed active/optimizer-visible count in the same established repo semantics; no arm-specific reserve is permitted and both reserve counts must be identical. Tests assert symmetry, not zero reserve.

- [ ] **Step 4: Implement information semantics**

`ChronologicalFailureArm.forward(...)` consumes only a canonical zero/null core token after contradiction and never accepts privileged core membership. `OracleConflictCoreArm.forward(...)` accepts `conflict_core_mask: Tensor[batch, variables]` only when `contradiction_observed=True`; otherwise raise `ValueError`.

Both arms execute `core_adapter` so neural compute/capacity remains matched. Oracle core conditioning uses the current mask only; no future-core or solution input exists in the interface.

- [ ] **Step 5: Implement analytical FLOP ledger**

Count event/variable projections, GRU steps, core-adapter, rollback head and verifier head. Emit per-arm max cost plus one `declared_max_accounted_flops_per_episode`. Set `hardware_profiler_flops_claimed=false`. Fail closed if required maximum exceeds a caller-supplied ceiling.

- [ ] **Step 6: Verify GREEN**

Run:

`pytest -q tests/test_matched_conflict_arms.py tests/test_matched_routing_arms.py tests/test_matched_cbrf_arms.py tests/test_matched_belief_arms.py`

---

### Task 3: Deterministic paired solvable conflict-world generator

**Files:**
- Create: `src/nolane_ai/experiments/exp286_conflict_worlds.py`
- Create: `tests/test_exp286_conflict_worlds.py`

**Interfaces:**
- Produces `Exp286ConflictBatch`.
- Produces `Exp286ConflictGenerator(root_seed: str).make_batch(...)`.
- Batch fields include `surface_events`, `variable_states`, `solution_targets`, `bad_branch_values`, `core_masks`, `decoy_order`, `metadata`, `digest`, `replicate`, `rng_stream`.
- Produces deterministic per-episode current-conflict metadata sufficient to materialize a core only after a bad branch is actually visited.

- [ ] **Step 1: Write RED generator tests**

For fixed `(root_seed, replicate, rng_stream, geometry)` require byte-identical tensors/metadata/digest across repeated calls. Changing replicate, stream or geometry must change digest. `rng_stream` must be restricted to `augmentation` or `evaluation`.

Assert every generated episode is globally solvable, has at least one deliberately bad local branch, has a non-empty exact current-core mask for that branch, and includes at least one decoy variable when `decoys>0`.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_exp286_conflict_worlds.py`.

- [ ] **Step 3: Implement generator with frozen lineage**

Seed only with:

```python
seed = derive_stream_seed(root_seed, "EXP-286", replicate, rng_stream)
```

Generate finite binary search episodes whose target assignment is known externally. Encode surface/variable tensors from the same world for both arms. The bad branch must be locally contradictory but the whole world must retain a verified productive solution path.

- [ ] **Step 4: Bind digest to raw bytes and metadata**

Use `tensor_raw_bytes` and `tensor_byteorder` exactly as EXP-279 does. Include experiment id, root seed, replicate, stream, seed, geometry, decoy order and contradiction/core metadata in the digest.

- [ ] **Step 5: Verify deterministic regeneration twice**

Run the generator tests twice; both runs must pass.

---

### Task 4: Paired search executor, training loop, DEVELOPMENT runner and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp286_paired_runner.py`
- Create: `tests/test_exp286_paired_runner.py`

**Interfaces:**
- Produces `run_exp286_paired_development(...) -> dict[str, Any]`.
- Produces `validate_exp286_paired_development(payload: dict[str, Any]) -> list[str]`.
- Produces `_artifact_digest(payload)` for canonical re-hash tests.
- Artifact schema exactly `NLM-EXP-286-PAIRED-DEV-EVAL-V1`.
- Arm order exactly `("chronological_failure", "oracle_conflict_core")`.

- [ ] **Step 1: Write RED scientific-boundary tests**

Require:

```python
assert payload["schema"] == "NLM-EXP-286-PAIRED-DEV-EVAL-V1"
assert payload["evidence_level"] == "EV-E2"
assert payload["decision"] == "UNVERIFIED"
assert payload["confirmatory_ready"] is False
assert payload["confirmatory_data_consumed"] is False
assert payload["challenge_materialized"] is False
assert payload["decision_rule_executed"] is False
assert payload["arm_order"] == ["chronological_failure", "oracle_conflict_core"]
assert payload["primary_endpoint"] == {
    "metric": "accounted_reasoning_flops_to_verified_solution",
    "direction": "lower",
    "mesi_relative_reduction": 0.15,
}
assert payload["protected_endpoints"]["verified_solution_rate_floor"] == (
    "oracle_conflict_core >= chronological_failure - 0.005"
)
```

Require initial functional digests match, training stream `augmentation`, evaluation stream `evaluation`, disjoint replicate ranges, unique paired world digests, and `validate_exp286_paired_development(payload) == []`.

- [ ] **Step 2: Write contradiction-time receipt and causal-divergence tests**

Top-level receipt must equal:

```python
{
  "artifact": "ground_truth_conflict_core",
  "ground_truth": True,
  "delivery_event": "after_current_contradiction_only",
  "delivered_to": ["oracle_conflict_core"],
  "withheld_from": ["chronological_failure"],
  "chronological_failure_received_conflict_core": False,
  "future_conflict_core_leakage": False,
  "solution_leakage": False,
}
```

Each raw row must retain one initial paired world digest but may record different per-arm visited branches/contradiction counts. Require `initial_world_pairing_closed=true`; do not require identical post-action paths.

- [ ] **Step 3: Write censoring and actual-path cost tests**

For every arm row require positive `accounted_reasoning_flops_to_verified_solution`, `verified_solution_rate` in `{0.0,1.0}`, and cost `<= declared_max_accounted_flops_per_episode`. If unresolved, require cost exactly equals ceiling and `censored_at_max_flops=true`; if solved, require `censored_at_max_flops=false` and an externally checked solution digest/flag.

- [ ] **Step 4: Write re-hashed semantic tamper tests**

Deep-copy a valid artifact, mutate one item, recompute `artifact_digest`, and require validator errors for each of:

- primary metric/direction/MESI;
- protected floor;
- arm order;
- top-level oracle receipt;
- row claiming chronological core receipt;
- row claiming oracle pre-contradiction delivery;
- future-core or solution leakage;
- parameter/optimizer-visible/resource flags;
- common compute ceiling;
- solved row with wrong verifier flag;
- unresolved row missing censoring or not charged at ceiling;
- removed/reordered raw replicate;
- training/evaluation overlap;
- aggregate drift;
- any confirmatory/challenge/promotion flag turned on.

- [ ] **Step 5: Implement seeded matched initialization and training**

Use:

```python
model_seed = derive_stream_seed(root_seed, "EXP-286", 0, "model_init")
```

Construct both arms inside `torch.random.fork_rng(devices=[])`, clone state dicts, and record identical functional state digests. Train both on the same ordered augmentation worlds and same optimizer hyperparameters. The training target is rollback/priority selection plus verifier support; oracle core is exposed only after synthetic current-contradiction events.

- [ ] **Step 6: Implement bounded search executor**

For each episode, start both arms from the same generated world and candidate order. At each step charge the analytical per-step cost. A branch can produce a current contradiction. Chronological receives the null core and rolls back chronologically; oracle receives only that current ground-truth core and uses its conditioned rollback scores. Preserve path divergence induced by the arm action. Stop when external target verification succeeds or common ceiling/max-search-steps is reached.

- [ ] **Step 7: Implement descriptive aggregate**

Report mean/median cost by arm, solution rates, paired mean log-cost ratio, descriptive relative FLOP reduction, censoring counts and sign consistency. Do not run confirmatory bootstrap CI and do not emit `PROMOTE_TO_NEXT_STAGE` or `KILL_SUBSYSTEM`.

- [ ] **Step 8: Implement validator as independent recomputation**

Recompute arm costs from row step receipts, censoring rule, aggregates, pair-audit digest, batch uniqueness/order and self-hash. Validation cannot simply trust stored booleans.

- [ ] **Step 9: Verify GREEN**

Run:

`pytest -q tests/test_exp286_paired_runner.py tests/test_matched_conflict_arms.py tests/test_exp286_conflict_worlds.py`

---

### Task 5: Neural Arm Registry integration without learned-localizer overclaim

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Modify: `tests/test_neural_arm_registry.py`
- Create: `tests/test_exp286_registry_integration.py`

**Interfaces:**
- Extend `TARGET_EXPERIMENTS` with `EXP-286`.
- Extend `_EXPECTED_ARMS` with frozen EXP-286 arm ids/descriptions.
- Extend `build_neural_arm_registry(..., exp286_pair_audit=None, exp286_execution_artifact=None)`.

- [ ] **Step 1: Extend test protocol fixture first**

Add exact frozen EXP-286 entry to `_protocol_subset()` before production registry changes, then assert default registry experiment set includes `EXP-286` and remains `match_court=BLOCKED`.

- [ ] **Step 2: Write RED registry evidence tests**

Valid pair audit must set both EXP-286 experiment-local arms to `IMPLEMENTED / MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM` while keeping scientific blockers. Valid execution must set `development_match_status="PAIRED_CONFLICT_HEADROOM_DEV_READY"` and retain `match_court="BLOCKED"`.

Require paired-execution evidence to record artifact/code digest, training/evaluation counts, descriptive relative FLOP reduction, solution-rate difference, censoring counts, and `learned_conflict_localizer_validated=false`.

- [ ] **Step 3: Write failure tests**

Reject invalid pair audit, execution without pair audit, tampered execution, protocol digest mismatch, or any execution that claims learned-localizer validation.

- [ ] **Step 4: Implement `_validate_exp286_pair_audit`**

Require exact schema, `EV-E2`, `UNVERIFIED`, total/functional/active/optimizer-visible parameter match, oracle information separation, common positive FLOP ceiling, two valid analytical ledgers, and `hardware_profiler_flops_claimed=false`.

- [ ] **Step 5: Implement registry state transitions**

Pair audit only → resource-matched development status. Pair audit + valid execution → `PAIRED_CONFLICT_HEADROOM_DEV_READY`. Remaining blockers must explicitly include confirmatory sample-size/analysis freeze, confirmatory-open execution and post-freeze challenge evidence. Never mark `ConflictCoreRegion` learned localization as validated.

- [ ] **Step 6: Verify GREEN and regressions**

Run EXP-286 registry tests plus `tests/test_neural_arm_registry.py`, EXP-277 and EXP-279 registry integration, and EXP-282 ceremony integration tests.

---

### Task 6: CLI, CI smoke, docs, version, review and exact-head merge

**Files:**
- Create: `scripts/run_exp286_paired_dev.py`
- Create: `tests/test_exp286_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`
- Modify: `README.md`

**Interfaces:**
- CLI writes one validated append-only EXP-286 DEVELOPMENT artifact and one registry snapshot.

- [ ] **Step 1: Write CLI RED tests**

Mirror the hardened EXP-279 CLI contract: tiny successful run, output overwrite refusal before protocol/model work, and rejection of a noncanonical protocol even when paired with a self-consistent local digest file.

- [ ] **Step 2: Implement CLI**

Before model creation:

1. refuse existing execution/registry outputs;
2. load and validate protocol;
3. compare file SHA to the committed `.sha256`;
4. call `require_canonical_stage_a_v1_digest`;
5. compute `source_tree_digest(ROOT)`;
6. run `run_exp286_paired_development`;
7. call `validate_exp286_paired_development` and abort on errors;
8. build registry using both pair audit and execution;
9. write both JSON artifacts only after validation.

- [ ] **Step 3: Wire full model-smoke coverage**

Add:

```text
tests/test_exp286_neural_import.py
tests/test_matched_conflict_arms.py
tests/test_exp286_conflict_worlds.py
tests/test_exp286_paired_runner.py
tests/test_exp286_registry_integration.py
tests/test_exp286_paired_cli.py
```

Then add one CPU-safe command:

```bash
python scripts/run_exp286_paired_dev.py \
  --tiny \
  --train-replicates 2 \
  --eval-replicates 3 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 5 \
  --decoys 2 \
  --max-search-steps 8 \
  --output /tmp/nlm-exp286-paired.json \
  --registry-output /tmp/nlm-exp286-neural-arm-registry.json
```

- [ ] **Step 4: Run all targeted tests before version bump**

Require EXP-286 tests green plus every existing EXP-277/279/282 model-smoke regression and `python -m compileall -q src scripts`.

- [ ] **Step 5: Version only after the lane is green**

Change `[project].version` and `nolane_ai.__version__` from `0.12.0` to `0.13.0` in the same verified head.

- [ ] **Step 6: Update README**

Document the EXP-286 command, contradiction-time-only oracle receipt, same-initial-world but causally divergent search paths, common censoring ceiling, and strict `EV-E2 / UNVERIFIED` boundary. State explicitly that oracle headroom does not validate learned `ConflictCoreRegion` localization.

- [ ] **Step 7: Exact-head verification**

Require GitHub Actions core 3.11, core 3.13 and model-smoke all green for the exact PR head. If any job fails, use systematic debugging before changing code.

- [ ] **Step 8: Frozen-protocol diff guard**

Compare PR against `main` and confirm neither `protocols/stage_a_v1.json` nor `protocols/stage_a_v1.sha256` changed.

- [ ] **Step 9: Reviewer pass**

Review exact diff for oracle timing/leakage, same-initial-world pairing, causal divergence preservation, parameter and optimizer-visible matching, actual-path compute accounting, censoring, semantic tamper resistance, registry scientific boundary, version synchronization and README wording. Resolve every Critical/Important issue.

- [ ] **Step 10: Squash merge with exact-head protection**

Merge only with `expected_head_sha` equal to the reviewed green head. Then verify the post-merge `main` workflow before claiming engineering closure.
