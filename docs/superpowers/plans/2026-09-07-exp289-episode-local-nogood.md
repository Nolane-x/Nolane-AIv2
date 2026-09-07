# EXP-289 Episode-Local Nogood Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an EV-E2 paired neural DEVELOPMENT lane for frozen `EXP-289` comparing `no_nogood` against exact `local_nogood` reuse under matched neural capacity, episode-local scope, deterministic paired restart worlds, independently audited over-prune safety, charged memory maintenance/retrieval cost, semantic tamper validation, registry/CLI/CI integration, and package `0.14.0`.

**Architecture:** Both arms use one identical experiment-local hybrid neural search envelope, identical initialization/world/restart lineage, and one common maximum accounted-cost ceiling. The only causal difference is that `local_nogood` may retain exact non-empty dead-end partial assignments observed on its own current-episode path and use exact subset matching on later states; `no_nogood` executes the matched null-memory neural path without persistence or pruning. Ground-truth valid-completion information belongs only to the evaluator/validator after the arm acts and must never gate insertion or feed the controller.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, existing NLM protocol/evidence/seed/tensor-byte/model-budget utilities, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-07-exp289-episode-local-nogood-design.md`

## Global Constraints

- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` are frozen and must not change.
- EXP-289 artifacts remain `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`, `confirmatory_data_consumed=false`, `challenge_seed_materialized=false`, `challenge_materialized=false`, `decision_rule_executed=false`.
- Frozen arms are exactly `no_nogood` and `local_nogood` with their exact frozen descriptions.
- Frozen primary endpoint is `repeat_dead_end_rate`, direction `lower`.
- Frozen protected endpoints are `valid_state_overprune_rate <= 0.005` and `local_nogood >= no_nogood - 0.01` on `verified_solution_rate`.
- Frozen MESI is relative RDER reduction `0.25`.
- Frozen analysis is paired relative RDER reduction with bootstrap CI plus explicit over-prune safety bound; DEVELOPMENT may emit descriptive statistics only and must not execute the scientific decision rule.
- Training uses the `augmentation` stream; DEVELOPMENT evaluation uses `evaluation`; model initialization uses `model_init`. No challenge or confirmatory randomness may be derived.
- Same replicate means same initial world, restart schedule, evaluator repeat-opportunity set, exogenous lineage, model initialization, and maximum episode budget. Post-action search divergence is scientific outcome and must be retained.
- The local store is exact and episode-scoped. No cross-episode/problem reuse, approximate retrieval, oracle conflict core, learned clause transfer, or lifelong lemma claim is permitted.
- Ground truth may audit stored nogoods only after arm action; it must not decide whether an arm inserts a nogood or feed any controller signal.
- The RDER denominator is arm-independent and predeclared by the generator/evaluator. Zero-opportunity episodes remain in raw evidence and are excluded from the RDER denominator only via the explicit schema rule.
- Storage, canonicalization, insertion, and retrieval/subset-comparison work must be charged to the accounted-cost ledger. Hardware-profiler FLOPs are never claimed.
- Scientific failures, over-prune violations, and ceiling exhaustion remain raw outcomes; they are never silently dropped or repaired by validator logic.
- Version stays `0.13.0` until the complete EXP-289 lane is GREEN; then both version authorities move together to `0.14.0`.

---

### Task 1: Establish a clean CI-visible RED contract

**Files:**
- Create: `tests/test_exp289_neural_import.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces CI-enforced imports for `build_matched_exp289_arm_pair` and `run_exp289_paired_development` before production modules exist.

- [ ] **Step 1: Write the failing import test**

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_exp289_neural_modules_import() -> None:
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development
    from nolane_ai.experiments.matched_nogood_arms import build_matched_exp289_arm_pair

    assert callable(run_exp289_paired_development)
    assert callable(build_matched_exp289_arm_pair)
```

- [ ] **Step 2: Add only this import test to model-smoke**

Append `tests/test_exp289_neural_import.py` to the explicit model-smoke pytest command. Do not create production EXP-289 modules in this commit.

- [ ] **Step 3: Commit and open the PR**

Commit the test plus workflow edit on `feat/exp289-episode-local-nogood`, then open a PR to `main` with the DEVELOPMENT scientific boundary stated explicitly.

- [ ] **Step 4: Verify intentional RED in GitHub Actions**

Expected: `core (3.11)` and `core (3.13)` stay GREEN; `model-smoke` fails only because `nolane_ai.experiments.exp289_paired_runner` and/or `matched_nogood_arms` does not yet exist. Existing tests must remain passing.

---

### Task 2: Matched neural arms and exact episode-scoped memory court

**Files:**
- Create: `src/nolane_ai/experiments/matched_nogood_arms.py`
- Create: `tests/test_matched_nogood_arms.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `NoNogoodArm(nn.Module)` and `LocalNogoodArm(nn.Module)` with identical trainable module inventories.
- Produces `Exp289ArmOutput` containing `branch_logits`, `verifier_confidence`, `memory_query_logit`, `memory_conditioning_used`, and `representation_semantics`.
- Produces `EpisodeScopedNogoodStore` with exact canonical subset matching and explicit scope metadata.
- Produces `build_matched_exp289_arm_pair(*, d_model: int, hidden_size: int, target_parameters: int, device=None) -> tuple[NoNogoodArm, LocalNogoodArm]`.
- Produces `audit_matched_exp289_arm_pair(..., restarts: int, variables: int, max_search_steps: int, max_accounted_cost_per_episode: int | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write RED parameter/state/resource tests**

Require identical state dictionaries immediately after construction and exact equality for total, functional, active-functional, and optimizer-visible parameter counts. Reserve capacity may exist only if required by the repo budget utility; reserve counts must be equal and must not be claimed as active/optimizer-visible capacity.

```python
no_nogood, local = build_matched_exp289_arm_pair(
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
)
audit = audit_matched_exp289_arm_pair(
    no_nogood,
    local,
    restarts=3,
    variables=6,
    max_search_steps=12,
)
assert audit["schema"] == "NLM-EXP-289-MATCHED-NOGOOD-ARMS-DEV-V1"
assert audit["parameter_match"] is True
assert audit["functional_parameter_match"] is True
assert audit["active_functional_parameter_match"] is True
assert audit["optimizer_visible_parameter_match"] is True
assert audit["memory_scope_closed"] is True
assert audit["compute_budget_closed"] is True
assert audit["hardware_profiler_flops_claimed"] is False
```

- [ ] **Step 2: Write RED exact-store semantics tests**

Require:

```python
store = EpisodeScopedNogoodStore(
    episode_digest="episode-a",
    problem_digest="problem-a",
)
store.add({"x": 0, "y": 1}, dead_end_observed=True)
assert store.matches({"x": 0, "y": 1, "z": 0}) is True
assert store.matches({"x": 0, "y": 0}) is False
```

Also require rejection of:
- empty/root insertion;
- `dead_end_observed=False` insertion;
- scope mismatch on query or reuse;
- cross-episode/problem import;
- any fuzzy/similarity API;
- duplicate insertion changing logical store size.

The store API must not accept `ground_truth_valid`, `oracle_conflict_core`, or any future-solution argument.

- [ ] **Step 3: Write RED arm information-separation tests**

`NoNogoodArm` must have no public API that accepts a real nogood hit/key as privileged input. Both arms execute the same neural memory-adapter path; baseline receives canonical null-memory response, local arm may receive only a boolean/current-query hit summary generated by the exact current-episode store. Output tensor shapes must match.

- [ ] **Step 4: Verify RED**

Run `pytest -q tests/test_matched_nogood_arms.py` and require missing module/symbol failures.

- [ ] **Step 5: Implement one shared trainable envelope**

Use one `_MatchedExp289ArmBase` with identical modules such as:

```python
self.event_projection = nn.Linear(d_model, hidden_size)
self.variable_projection = nn.Linear(d_model, hidden_size)
self.deliberation_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
self.memory_adapter = nn.Linear(1, hidden_size)
self.branch_head = nn.Linear(hidden_size, 1)
self.verifier_head = nn.Linear(hidden_size, 1)
self.mix_gate = nn.Parameter(torch.zeros(()))
finalize_region_budget(self, target_parameters, device=device, frozen=False)
```

Use established repo helpers for functional/optimizer-visible accounting. Build both arms from the same seeded template state so their functional initialization digests are identical.

- [ ] **Step 6: Implement the exact scoped store**

Canonicalize a non-empty assignment as sorted `(variable, int(value))` pairs and store `frozenset` keys. Matching is exact subset inclusion only. Scope is constructor-bound and immutable for the store lifetime. Track insertion count, query count, comparison count, hit count, and canonicalization operations for later charged-cost receipts.

- [ ] **Step 7: Implement the analytical maximum-cost court**

Emit a shared declared maximum per-episode cost and per-arm neural/memory components. The local arm maximum includes worst-case canonicalization, insertions, and subset comparisons; the baseline maximum includes the same neural adapter path but no fabricated store activity. Fail closed if caller-supplied ceiling is below either maximum.

- [ ] **Step 8: Verify GREEN plus prior arm regressions**

Run:

`pytest -q tests/test_matched_nogood_arms.py tests/test_matched_conflict_arms.py tests/test_matched_routing_arms.py tests/test_matched_cbrf_arms.py tests/test_matched_belief_arms.py`

---

### Task 3: Deterministic paired restart-world and opportunity generator

**Files:**
- Create: `src/nolane_ai/experiments/exp289_nogood_worlds.py`
- Create: `tests/test_exp289_nogood_worlds.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `Exp289RestartWorld` / `Exp289RestartBatch` dataclasses.
- Produces `Exp289NogoodGenerator(root_seed: str).make_batch(...)`.
- Batch binds finite-domain problem tensors/metadata, valid solutions, restart schedules, evaluator-only canonical repeat-opportunity metadata, lineage, and raw-byte digest.
- Produces `regenerate_exp289_world(...)` for validator reconstruction.

- [ ] **Step 1: Write RED determinism/lineage tests**

For identical `(root_seed, replicate, rng_stream, geometry)`, require byte-identical tensors, restart schedule, evaluator opportunity metadata, valid-solution set, and digest. Changing replicate, stream, restart count, variables, or decoy geometry must change the bound digest.

`rng_stream` must accept `augmentation` and `evaluation` only for world generation.

- [ ] **Step 2: Write RED world semantics tests**

Each episode must be globally solvable and expose at least one arm-independent repeat opportunity under the shared restart schedule in non-degenerate test geometry. Require:
- at least one valid solution;
- at least one non-empty dead-end partial assignment;
- a later restart in which the same canonical logical dead-end class can be re-entered;
- deterministic variable/domain order perturbations;
- evaluator opportunity metadata not included in arm input tensors.

- [ ] **Step 3: Write RED zero-opportunity edge-case test**

Generator may emit an explicitly marked zero-opportunity episode for a special tiny geometry, but it must remain in raw evidence with `rder_denominator_eligible=false`; it may never disappear from batch length or lineage digests.

- [ ] **Step 4: Implement by-construction solvable restart worlds**

Use existing `CanonicalProblemState`/constraint primitives. Construct a known productive solution path plus decoy/contradictory branches whose canonical partial dead ends can recur under deterministic restart order permutations. Compute valid solutions/evaluator opportunity classes outside arm inputs.

Seed only with:

```python
seed = derive_stream_seed(root_seed, "EXP-289", replicate, rng_stream)
```

Bind raw tensor bytes plus problem/restart/opportunity metadata into the batch digest.

- [ ] **Step 5: Verify GREEN**

Run `pytest -q tests/test_exp289_nogood_worlds.py` and then the existing conflict/routing world-generator tests.

---

### Task 4: Paired DEVELOPMENT runner, search executor, cost receipts, and semantic validator

**Files:**
- Create: `src/nolane_ai/experiments/exp289_paired_runner.py`
- Create: `tests/test_exp289_paired_runner.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces `EXP289_ARTIFACT_SCHEMA = "NLM-EXP-289-PAIRED-DEV-EVAL-V1"`.
- Produces `run_exp289_paired_development(...) -> dict[str, Any]`.
- Produces `validate_exp289_execution_artifact(artifact: dict[str, Any], protocol: dict[str, Any]) -> None`.
- Produces deterministic episode execution with raw restart/search/dead-end/store/query/cost receipts.

- [ ] **Step 1: Write RED top-level scientific-boundary tests**

Require artifact fields:

```python
assert artifact["schema"] == "NLM-EXP-289-PAIRED-DEV-EVAL-V1"
assert artifact["evidence_level"] == "EV-E2"
assert artifact["decision"] == "UNVERIFIED"
assert artifact["confirmatory_ready"] is False
assert artifact["confirmatory_data_consumed"] is False
assert artifact["challenge_seed_materialized"] is False
assert artifact["challenge_materialized"] is False
assert artifact["decision_rule_executed"] is False
```

Require exact frozen arms/endpoints/MESI/protected-floor receipts and exact protocol digest.

- [ ] **Step 2: Write RED episode-scope/action-order tests**

For every local-store insertion require raw evidence that:
1. the partial state was actually reached;
2. a dead end was observed first;
3. key is non-empty and canonical;
4. scope digests match the current episode/problem;
5. no oracle/future/ground-truth gate signal was delivered to the arm before insertion.

Ground-truth `no_valid_completion` may appear only in evaluator audit fields written after the arm action.

- [ ] **Step 3: Write RED RDER denominator/numerator tests**

Recompute from raw rows:

```text
repeat_dead_end_rate = repeated_dead_end_reentries / predeclared_repeat_opportunities
```

Denominator must match generator-regenerated arm-independent opportunity receipts. Prevented repeats may reduce the numerator but never denominator. Zero-opportunity episodes remain raw and are explicitly excluded from the aggregate denominator.

- [ ] **Step 4: Write RED over-prune safety tests**

Validator must regenerate the world and valid solution/prefix set and recompute whether every stored nogood intersects any valid completion path. A positive over-prune event is a scientific result and must remain accepted when truthfully represented; validator rejects only forged/deleted/inconsistent safety evidence.

Add tamper tests that:
- change a stored key to match a valid solution and falsely leave over-prune at zero;
- delete an offending row;
- alter only the aggregate rate;
then recompute the top-level self-hash. Validator must still reject all three.

- [ ] **Step 5: Write RED cost/censoring tests**

Recompute per-step and per-episode:
- shared neural FLOPs;
- memory-adapter FLOPs;
- canonicalization operations;
- insertion operations;
- subset comparison/retrieval operations;
- total `accounted_reasoning_cost`.

Assert local-store work is charged. If the common ceiling is exhausted, row remains present, solution failure is retained, cost equals ceiling, and `censored_at_max_cost=true`.

- [ ] **Step 6: Write RED lineage/training separation tests**

Training replicate range uses `augmentation`; development evaluation uses disjoint `evaluation` replicate range; model initialization uses `model_init`. Reject any overlap or use of challenge/confirmatory-derived randomness.

- [ ] **Step 7: Write RED semantic-tamper matrix**

After re-hashing, validator must reject drift in:
- protocol/arm descriptions;
- endpoint/MESI/protected floors;
- initial-state/model-init digests;
- parameter audit;
- restart schedule/opportunity set;
- episode/problem scope;
- empty/root or cross-scope nogoods;
- insertion-before-dead-end order;
- approximate matching claims;
- repeated-dead-end numerator;
- prevented-repeat/nogood-hit counts;
- candidate/verified solution receipts;
- memory cost accounting;
- ceiling/censoring;
- cross-episode/learned-clause/lifelong-lemma overclaims;
- confirmatory/challenge/promotion flags;
- blockers;
- artifact self-hash.

- [ ] **Step 8: Implement minimal paired training loop**

Use identical initial arm states and optimizer family/hyperparameters/update count. Training may shape the shared neural substrate but cannot introduce cross-episode learned memory. Store lifecycle resets per episode.

- [ ] **Step 9: Implement deterministic paired evaluation executor**

For each paired episode/restart:
1. reconstruct shared initial world/order;
2. query current local store only for `local_nogood`;
3. run the matched neural step;
4. execute branch/search action;
5. record reached state and independently observable consistency/dead-end event;
6. let the local arm insert the observed non-empty exact dead-end key without ground-truth gating;
7. only afterward run evaluator valid-completion audit for scientific safety evidence;
8. accumulate raw neural/memory cost receipts;
9. retain causal divergence and all failures.

- [ ] **Step 10: Implement descriptive aggregate only**

Report arm RDER, relative descriptive reduction, verified solution rate, valid-state over-prune rate, memory hit/store counts, and accounted-cost summaries. Do not calculate or execute the frozen confirmatory bootstrap decision rule.

- [ ] **Step 11: Implement reconstruction-first validator**

Validator must regenerate model-init expectations, matched pair audit, worlds/restart schedules/opportunity denominators, raw valid-completion safety truth, RDER aggregates, and cost/censoring from sealed lineage/config. It must not trust artifact booleans merely because the self-hash is valid.

- [ ] **Step 12: Verify GREEN**

Run:

`pytest -q tests/test_exp289_paired_runner.py tests/test_exp289_nogood_worlds.py tests/test_matched_nogood_arms.py`

Then run EXP-286/279/282 paired-runner regression tests.

---

### Task 5: Neural-arm registry integration without scientific overclaim

**Files:**
- Modify: `src/nolane_ai/experiments/neural_arm_registry.py`
- Create: `tests/test_exp289_registry_integration.py`
- Modify: `tests/test_neural_arm_registry.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Extend `TARGET_EXPERIMENTS` with `EXP-289`.
- Extend frozen exact arm-description map with `no_nogood` and `local_nogood`.
- Add optional `exp289_pair_audit` and `exp289_execution_artifact` inputs to registry construction.
- A valid pair + execution may advance engineering status to `PAIRED_LOCAL_NOGOOD_DEV_READY`; `match_court` remains `BLOCKED`.

- [ ] **Step 1: Write RED registry tests**

Require EXP-289 in target set and exact frozen descriptions. With valid pair audit + execution artifact require:

```python
entry = registry["experiments"]["EXP-289"]
assert entry["development_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
assert entry["match_court"] == "BLOCKED"
assert entry["evidence_level"] == "EV-E2"
assert entry["decision"] == "UNVERIFIED"
```

- [ ] **Step 2: Write RED failure/overclaim tests**

Registry must reject:
- pair-only evidence with scope/protocol/parameter mismatch;
- execution artifact failing the EXP-289 validator;
- forged cross-episode reuse;
- `learned_clause_transfer_validated=true`;
- `lifelong_lemma_economy_validated=true`;
- `confirmatory_ready=true`;
- missing/deleted blocker fields;
- forged RDER/over-prune summaries.

- [ ] **Step 3: Verify RED**

Run `pytest -q tests/test_exp289_registry_integration.py tests/test_neural_arm_registry.py`.

- [ ] **Step 4: Implement validator-backed registry evidence**

Call the EXP-289 pair/execution validators rather than trusting readiness flags. Record raw evidence summaries such as RDER denominator-eligible count, repeated reentries, prevented repeats, nogood hits/stores, over-prune event count, censored episode count, and accounted-cost ceiling.

- [ ] **Step 5: Verify GREEN and all prior registry regressions**

Run:

`pytest -q tests/test_exp289_registry_integration.py tests/test_neural_arm_registry.py tests/test_exp286_registry_integration.py tests/test_exp279_registry_integration.py tests/test_exp277_registry_integration.py`

---

### Task 6: Hardened CLI, CI smoke, documentation, version, exact-head review, and merge

**Files:**
- Create: `scripts/run_exp289_paired_dev.py`
- Create: `tests/test_exp289_paired_cli.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `src/nolane_ai/__init__.py`

**Interfaces:**
- CLI accepts tiny/default DEVELOPMENT geometry and optional registry output.
- CLI verifies canonical frozen protocol authority before execution.
- CLI refuses overwrite before model execution.
- CLI validates execution and registry artifacts before writing.

- [ ] **Step 1: Write RED CLI tests and CI wiring before the script exists**

Require successful tiny invocation shape such as:

```bash
python scripts/run_exp289_paired_dev.py \
  --tiny \
  --train-replicates 2 \
  --eval-replicates 3 \
  --batch-size 2 \
  --restarts 3 \
  --variables 6 \
  --decoys 2 \
  --max-search-steps 12 \
  --output /tmp/nlm-exp289-paired.json \
  --registry-output /tmp/nlm-exp289-neural-arm-registry.json
```

Tests also require overwrite refusal before runner invocation, canonical protocol digest rejection on mismatch, output boundary flags all false, valid self-hashes, and registry `match_court=BLOCKED`.

Add all EXP-289 test files and this tiny CLI command to model-smoke while the script is still absent; capture the intended RED due only to the missing script.

- [ ] **Step 2: Implement fail-closed CLI order**

The script must execute in this order:
1. reject existing output/registry paths;
2. load canonical `protocols/stage_a_v1.json` and verify `.sha256`;
3. compute source-tree digest;
4. run paired DEVELOPMENT geometry;
5. call `validate_exp289_execution_artifact`;
6. optionally build and validate the neural-arm registry;
7. only then atomically write JSON outputs.

No challenge/confirmatory seed derivation exists in the CLI.

- [ ] **Step 3: Verify CLI GREEN in CI**

Require model-smoke to complete pytest, audits, training/evaluation smoke, EXP-277, EXP-279, EXP-286, EXP-289, EXP-282 CLI regressions, confirmatory-prep smoke, and compileall.

- [ ] **Step 4: Bump release only after full lane is GREEN**

Change both:
- `pyproject.toml`: `version = "0.14.0"`;
- `src/nolane_ai/__init__.py`: `__version__ = "0.14.0"`.

- [ ] **Step 5: Document the DEVELOPMENT boundary**

README must state:
- exact frozen arms;
- episode-local exact subset memory only;
- evaluator truth does not gate arm insertion;
- arm-independent predeclared RDER denominator;
- over-prune and solution-rate safety metrics;
- charged storage/retrieval cost;
- challenge/confirmatory fields remain false;
- `EV-E2 / UNVERIFIED` and `match_court=BLOCKED`;
- no EXP-290 transfer or lifelong lemma claim.

- [ ] **Step 6: Run final exact-head verification**

GitHub Actions exact PR head must show `core (3.11)=success`, `core (3.13)=success`, `model-smoke=success`. Confirm the frozen protocol files are absent from the PR changed-file set.

- [ ] **Step 7: Reviewer/self-review gate**

Review exact base→head diff for Critical/Important issues, especially:
- ground-truth leakage into arm actions;
- RDER denominator manipulation;
- unsound/cross-scope store reuse;
- hidden memory cost;
- truthful acceptance of negative over-prune outcomes;
- validator self-consistency instead of independent reconstruction;
- overclaims beyond EV-E2.

Inspect PR issue comments, review submissions, and inline review threads. Resolve every Critical/Important issue before merge.

- [ ] **Step 8: Squash merge with race protection**

Re-read PR head SHA and `main` SHA immediately before merge. Squash merge only with the exact expected PR head. Do not merge if base/head moved unexpectedly.

- [ ] **Step 9: Verify post-merge `main` CI**

Wait for the push-triggered CI on the resulting squash commit and require all three jobs GREEN before claiming EXP-289 engineering closure.

---

## Acceptance Checklist

Implementation is complete only when all of the following are true:

1. A clean CI-visible TDD RED exists before EXP-289 production modules.
2. Exact total/functional/active-functional/optimizer-visible parameter matching passes.
3. Both arms have identical functional initialization and neural module inventory.
4. Local memory is exact, non-empty, episode/problem scoped, and uses subset matching only.
5. No arm receives ground-truth valid-completion, oracle conflict-core, future-solution, or learned-transfer information.
6. Evaluator ground truth audits store actions only after the actions occur.
7. Deterministic paired worlds/restart schedules/opportunity sets regenerate byte-identically from sealed lineage.
8. RDER denominator is arm-independent and cannot be path-manipulated.
9. Zero-opportunity episodes remain in raw evidence under an explicit exclusion rule.
10. Over-prune safety is independently recomputed and truthful negative outcomes remain valid scientific evidence.
11. Store canonicalization/insertion/query/comparison cost is charged and reconstructed from raw receipts.
12. Ceiling exhaustion remains a censored scientific outcome rather than a dropped row.
13. Semantic tamper tests fail closed after self-hash recomputation.
14. Registry cannot clear confirmatory blockers or claim EXP-290/lifelong-lemma validation.
15. Tiny EXP-289 CLI passes in GitHub Actions with all prior experiment regressions.
16. Frozen Stage-A protocol files are absent from the PR diff.
17. Package/runtime versions agree at `0.14.0` only after full lane GREEN.
18. Exact-head core 3.11, core 3.13, and model-smoke CI are GREEN.
19. Review has no unresolved Critical/Important issue.
20. Squash merge uses exact expected-head protection and post-merge `main` CI is GREEN before engineering closure is claimed.
