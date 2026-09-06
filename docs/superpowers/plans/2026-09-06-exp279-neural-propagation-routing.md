# EXP-279 Neural Propagation Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen EXP-279 comparing `propagation_only`, `branch_only`, and `hybrid` under matched parameters, reclaimed-capacity accounting, equal max accounted FLOPs, and predeclared structure-fit strata.

**Architecture:** Reuse the EXP-277 experiment-local neural substrate and structure-dense generator patterns. Add a three-arm matched execution module, deterministic stratified generator, self-validating paired runner, registry evidence gating, and CPU-safe CLI while preserving Stage-A V1 protocol bytes and keeping all scientific claims `UNVERIFIED`.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed/resource utilities.

**Spec:** `docs/superpowers/specs/2026-09-06-exp279-neural-propagation-routing-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-279 development artifacts are `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`.
- `confirmatory_data_consumed=false`, `challenge_materialized=false`, `decision_rule_executed=false`.
- Frozen arm IDs are exactly `propagation_only`, `branch_only`, `hybrid`.
- Frozen primary metric is exactly `verified_utility_per_accounted_flop_on_structure_dense_stratum`, higher is better.
- Frozen MESI is relative gain `0.08`.
- Frozen protected floor is exactly `hybrid >= best_simple - 0.01` on verified solution rate.
- Frozen multiplicity family is `PROPAGATION_ROUTING`.
- Training RNG stream is `augmentation`; development evaluation RNG stream is `evaluation`.
- Structure-fit strata are exactly `PROPAGATION_FRIENDLY`, `BRANCH_FRIENDLY`, `MIXED_STRUCTURE`.
- No confirmatory Holm inference is executed in this PR.
- Simpler rivals must receive active reclaimed capacity rather than hidden excluded reserve.
- All three arms share one declared max accounted-FLOP ceiling.

---

### Task 1: Establish EXP-279 RED import contract

**Files:**
- Create: `tests/test_exp279_neural_import.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: frozen EXP-279 arm IDs from Stage-A V1.
- Produces: CI-enforced expectation for `matched_routing_arms` and `exp279_paired_runner`.

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

- [ ] **Step 2: Add only `tests/test_exp279_neural_import.py` to explicit model-smoke**

Keep this first commit intentionally RED; do not add production modules yet.

- [ ] **Step 3: Push and preserve GitHub RED**

Expected: core jobs green; model-smoke fails only because `nolane_ai.experiments.exp279_paired_runner` or `matched_routing_arms` does not exist.

- [ ] **Step 4: Commit**

```bash
git add tests/test_exp279_neural_import.py .github/workflows/ci.yml
git commit -m "test: require EXP-279 neural routing modules"
```

---

### Task 2: Matched three-arm neural execution and reclaimed-capacity audit

**Files:**
- Create: `src/nolane_ai/experiments/matched_routing_arms.py`
- Create: `tests/test_matched_routing_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces:
  - `PropagationOnlyArm`
  - `BranchOnlyArm`
  - `HybridRoutingArm`
  - `Exp279ArmOutput`
  - `build_matched_exp279_arm_triplet(...)`
  - `audit_matched_exp279_arm_triplet(...)`
- All arm outputs contain `decision_logits`, `verifier_confidence`, `representation_semantics`, and an execution receipt.

- [ ] **Step 1: Write behavior tests before production code**

Tests must instantiate a tiny triplet and assert:

```python
propagation, branch, hybrid = build_matched_exp279_arm_triplet(
    d_model=8,
    hidden_size=8,
    target_parameters=4_000,
)

audit = audit_matched_exp279_arm_triplet(
    propagation,
    branch,
    hybrid,
    timesteps=3,
    variables=4,
    constraints=2,
)

assert audit["parameter_match"] is True
assert audit["functional_parameter_match"] is True
assert audit["optimizer_visible_parameter_match"] is True
assert audit["reclaimed_capacity_active"] is True
assert audit["inactive_excluded_reclaimed_parameters"] == 0
assert audit["compute_budget_closed"] is True
assert set(audit["compute_ledger"]) >= {"propagation_only", "branch_only", "hybrid_stop", "hybrid_branch"}
```

Also test:

- `branch_only(surface, variables, incidence=...)` raises `TypeError` or `ValueError`.
- propagation requires rank-3 incidence.
- hybrid requires rank-3 incidence.
- all three decision logits have identical `[batch, variables, 2]` shape.
- all three initial functional digests match after builder cloning.
- `capacity_reserve` is not used to hide reclaimed parameters.
- both hybrid paths are below the same declared ceiling.
- every ledger has `hardware_profiler_flops_claimed=false`.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_matched_routing_arms.py
```

Expected: missing module/classes.

- [ ] **Step 3: Implement the shared matched base**

Use a shared module layout so state dictionaries are shape-identical:

```python
class _MatchedExp279Base(nn.Module):
    def __init__(self, d_model, hidden_size, target_parameters, *, device=None):
        super().__init__()
        self.event_projection = nn.Linear(d_model, hidden_size, device=device)
        self.variable_projection = nn.Linear(d_model, hidden_size, device=device)
        self.branch_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, device=device)
        self.propagation_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.reclaimed_projection = nn.Linear(hidden_size, hidden_size, device=device)
        self.routing_head = nn.Linear(hidden_size, 1, device=device)
        self.decision_head = nn.Linear(hidden_size, 2, device=device)
        self.verifier_head = nn.Linear(hidden_size, 1, device=device)
        finalize_region_budget(self, target_parameters, device=device, frozen=False)
```

Do not exclude `reclaimed_projection` from `functional_trainable_named_parameters` or optimizer groups.

- [ ] **Step 4: Implement arm semantics**

`PropagationOnlyArm`:

- uses incidence propagation
- does not execute `branch_gru`
- routes reclaimed branch capacity through `reclaimed_projection`

`BranchOnlyArm`:

- signature contains no incidence argument
- executes recurrent branch encoding
- routes reclaimed propagation capacity through `reclaimed_projection`

`HybridRoutingArm`:

- propagation first
- residual uncertainty = mean variable entropy proxy from propagation logits or a deterministic hidden-state proxy fixed before evaluation
- branch executes iff residual uncertainty exceeds constructor threshold
- receipt records `branch_executed`, `residual_uncertainty`, `routing_threshold`

- [ ] **Step 5: Implement analytical compute ledgers**

Return separate costs for:

- shared encoding
- propagation
- branch recurrence
- reclaimed-capacity pathway
- routing head
- decision/verifier heads

Hybrid has two declared path ledgers: `hybrid_stop` and `hybrid_branch`. Fail if any exceeds `declared_max_accounted_flops_per_episode`.

- [ ] **Step 6: Verify GREEN and EXP-277 regression**

Run:

```bash
pytest -q tests/test_matched_routing_arms.py tests/test_matched_cbrf_arms.py
```

- [ ] **Step 7: Commit**

```bash
git add src/nolane_ai/experiments/matched_routing_arms.py tests/test_matched_routing_arms.py .github/workflows/ci.yml
git commit -m "feat: add matched EXP-279 routing arms"
```

---

### Task 3: Deterministic predeclared structure-fit generator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_structure_strata.py`
- Create: `tests/test_exp279_structure_strata.py`

**Interfaces:**
- Produces `Exp279StructureBatch` and `Exp279StructureStrataGenerator.make_batch(...)`.
- `make_batch` accepts explicit `stratum` and returns paired tensors plus deterministic digest.

- [ ] **Step 1: Write determinism and semantics tests**

For each of the three strata, assert same seed/replicate/stream/geometry regenerates byte-identical tensors and digest. Assert changing replicate, stream, stratum, or geometry changes digest.

Assert:

```python
assert batch.structure_fit_stratum in {
    "PROPAGATION_FRIENDLY",
    "BRANCH_FRIENDLY",
    "MIXED_STRUCTURE",
}
assert batch.incidence.ndim == 3
assert batch.targets.ndim == 2
assert set(batch.targets.unique().tolist()) <= {0, 1}
```

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_exp279_structure_strata.py
```

- [ ] **Step 3: Implement generator**

Use only:

```python
seed = derive_stream_seed(root_seed, "EXP-279", replicate, rng_stream)
```

Then derive the explicit stratum geometry deterministically from the local generator. Encode stratum into digest metadata so the same replicate under a different stratum cannot collide.

Suggested DEVELOPMENT geometry:

- propagation-friendly: more variables per constraint component and lower observation noise
- branch-friendly: more components relative to variables, sparser incidence, higher ambiguity/noise
- mixed: intermediate geometry

All strata still obey caller-provided maximum `variables` / `constraints`; tests validate shape consistency rather than empirical arm superiority.

- [ ] **Step 4: Verify GREEN twice**

```bash
pytest -q tests/test_exp279_structure_strata.py
pytest -q tests/test_exp279_structure_strata.py
```

- [ ] **Step 5: Commit**

```bash
git add src/nolane_ai/experiments/exp279_structure_strata.py tests/test_exp279_structure_strata.py
git commit -m "feat: add EXP-279 stratified paired worlds"
```

---

### Task 4: Paired DEVELOPMENT runner and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp279_paired_runner.py`
- Create: `tests/test_exp279_paired_runner.py`

**Interfaces:**
- Produces:
  - `run_exp279_paired_development(...) -> dict[str, Any]`
  - `validate_exp279_paired_development(payload: dict[str, Any]) -> list[str]`
- Artifact schema: `NLM-EXP-279-PAIRED-DEV-EVAL-V1`.

- [ ] **Step 1: Write runner boundary tests**

Assert generated artifact fields:

```python
assert artifact["schema"] == "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
assert artifact["evidence_level"] == "EV-E2"
assert artifact["decision"] == "UNVERIFIED"
assert artifact["confirmatory_ready"] is False
assert artifact["confirmatory_data_consumed"] is False
assert artifact["challenge_materialized"] is False
assert artifact["decision_rule_executed"] is False
assert artifact["primary_endpoint"] == {
    "metric": "verified_utility_per_accounted_flop_on_structure_dense_stratum",
    "direction": "higher",
    "mesi_relative_gain": 0.08,
}
assert artifact["protected_endpoints"]["verified_solution_rate_floor"] == "hybrid >= best_simple - 0.01"
assert artifact["multiplicity_family"] == "PROPAGATION_ROUTING"
```

Also assert exact three strata and `augmentation`/`evaluation` lineage separation.

- [ ] **Step 2: Write raw-row and hybrid path tests**

Each evaluation row must bind:

- replicate
- structure-fit stratum
- paired batch digest
- all three arm metrics
- `world_pairing_closed=true`
- hybrid routing receipt
- hybrid charged cost equal to the ledger path selected by `branch_executed`

- [ ] **Step 3: Write rehash-tamper tests**

Deep-copy the artifact, mutate one semantic field, recompute top-level `artifact_digest`, and verify validator still rejects each mutation:

- arm order
- MESI 0.08
- protected floor
- multiplicity family
- stratum list/order
- branch-only information receipt
- reclaimed-capacity match
- compute ceiling
- training/evaluation stream
- evaluation ordering
- hybrid branch receipt/cost mismatch
- confirmatory/challenge flags

- [ ] **Step 4: Verify RED**

```bash
pytest -q tests/test_exp279_paired_runner.py
```

- [ ] **Step 5: Implement runner**

Build one seeded triplet from `model_init`; clone identical functional initialization. Train all arms on exactly the same `augmentation` batches. Cycle strata deterministically in the fixed order:

```python
STRATA = (
    "PROPAGATION_FRIENDLY",
    "BRANCH_FRIENDLY",
    "MIXED_STRUCTURE",
)
```

Evaluate on disjoint `evaluation` replicates. External metrics use exact target correctness and charged path FLOPs.

Primary metric per arm is:

```python
verified_solution_rate / max(accounted_flops_per_episode, 1)
```

stored under the exact frozen metric name. Descriptive relative gains may be emitted but never a confirmatory decision.

- [ ] **Step 6: Implement validator**

Validator recomputes aggregates from raw rows and checks top-level self-hash with `canonical_sha256`. It validates the fixed authority constants directly, rather than trusting mutable artifact metadata.

- [ ] **Step 7: Verify GREEN**

```bash
pytest -q tests/test_exp279_paired_runner.py tests/test_matched_routing_arms.py tests/test_exp279_structure_strata.py
```

- [ ] **Step 8: Commit**

```bash
git add src/nolane_ai/experiments/exp279_paired_runner.py tests/test_exp279_paired_runner.py
git commit -m "feat: add EXP-279 paired development runner"
```

---

### Task 5: Neural-arm registry evidence gating

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Create: `tests/test_exp279_registry_integration.py`
- Modify: `tests/test_neural_arm_registry.py` only if shared signature assertions require it.

**Interfaces:**
- Extend:

```python
build_neural_arm_registry(
    *,
    protocol,
    protocol_digest,
    model_audit,
    exp277_pair_audit=None,
    exp277_execution_artifact=None,
    exp279_triplet_audit=None,
    exp279_execution_artifact=None,
    exp282_pair_audit=None,
    exp282_execution_artifact=None,
)
```

- [ ] **Step 1: Write failing integration tests**

Cases:

1. valid triplet audit marks all EXP-279 arms `IMPLEMENTED` with `MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM`.
2. valid paired execution produces `development_match_status="PAIRED_STRATIFIED_ROUTING_DEV_READY"`.
3. `match_court` remains `BLOCKED`.
4. execution without triplet audit raises.
5. invalid semantic artifact raises.
6. protocol digest mismatch raises.
7. compute-budget-open audit raises.
8. inactive/excluded reclaimed capacity raises.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_exp279_registry_integration.py
```

- [ ] **Step 3: Implement EXP-279 validation and registry wiring**

Validate schema/evidence/parameter match/reclaimed-capacity/compute ledger before accepting evidence. Preserve existing EXP-277/282 behavior unchanged.

With execution evidence, blockers become only scientific closure debts such as:

- confirmatory blocked sample-size/analysis freeze
- confirmatory-open execution
- post-freeze challenge

Do not set `match_court` green.

- [ ] **Step 4: Verify GREEN regressions**

```bash
pytest -q tests/test_exp279_registry_integration.py tests/test_neural_arm_registry.py tests/test_exp277_registry_integration.py
```

- [ ] **Step 5: Commit**

```bash
git add src/nolane_ai/experiments/neural_arm_registry.py tests/test_exp279_registry_integration.py tests/test_neural_arm_registry.py
git commit -m "feat: register EXP-279 development evidence"
```

---

### Task 6: CLI, CI closure, docs, and package 0.12.0

**Files:**
- Create: `scripts/run_exp279_paired_dev.py`
- Create: `tests/test_exp279_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`
- Modify: `README.md`

**Interfaces:**
- CLI writes one EXP-279 artifact and optional registry artifact; both paths are append-only/refuse overwrite.

- [ ] **Step 1: Write CLI RED tests**

Test:

- tiny successful execution
- output refusal before experiment execution if artifact path exists
- registry-output refusal before execution if registry path exists
- canonical frozen protocol verification
- `EV-E2 / UNVERIFIED` boundary
- no confirmatory/challenge material
- semantic validator passes generated artifact

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_exp279_paired_cli.py
```

- [ ] **Step 3: Implement CLI**

Use existing Stage-A protocol loader/verifier and source-tree digest utility. Tiny defaults should be CPU-safe, for example:

- target parameters: small experiment-local envelope
- train replicates: 1
- eval replicates: 3 so every stratum appears once
- batch size: 2
- timesteps: 3
- variables: 4
- constraints: 2

Default non-tiny settings may be larger DEVELOPMENT geometry but must not imply full 16M matching.

- [ ] **Step 4: Expand model-smoke only after unit GREEN**

Add:

- `tests/test_matched_routing_arms.py`
- `tests/test_exp279_structure_strata.py`
- `tests/test_exp279_paired_runner.py`
- `tests/test_exp279_registry_integration.py`
- `tests/test_exp279_paired_cli.py`

Then add one tiny command:

```bash
python scripts/run_exp279_paired_dev.py \
  --tiny \
  --train-replicates 1 \
  --eval-replicates 3 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 4 \
  --constraints 2 \
  --output /tmp/nlm-exp279-paired.json \
  --registry-output /tmp/nlm-exp279-neural-arm-registry.json
```

- [ ] **Step 5: Version only after complete EXP-279 GREEN**

Set both:

```toml
version = "0.12.0"
```

and:

```python
__version__ = "0.12.0"
```

- [ ] **Step 6: Document bounded status**

README must explicitly say EXP-279 lane is DEVELOPMENT `EV-E2 / UNVERIFIED`, analytical FLOPs are not hardware profiler claims, confirmatory Holm analysis is not run, and `match_court` remains blocked.

- [ ] **Step 7: Full PR verification**

Run the exact CI matrix and confirm:

- core 3.11 green
- core 3.13 green
- model-smoke green
- EXP-277 regression green
- EXP-282 ceremony regression green
- tiny EXP-279 CLI green
- compileall green
- `protocols/stage_a_v1.json` absent from diff
- `protocols/stage_a_v1.sha256` absent from diff

- [ ] **Step 8: Reviewer pass**

Review specifically:

- branch-only cannot receive incidence
- reclaimed capacity is active/optimizer-visible
- hybrid branch path cost cannot be undercharged
- structure strata are fixed before evaluation
- semantic validator catches rehashed tampering
- no confirmatory Holm decision leaks into DEVELOPMENT

- [ ] **Step 9: Squash merge exact green head**

Use expected-head SHA when merging. After merge, verify push CI on `main` before claiming closure.
