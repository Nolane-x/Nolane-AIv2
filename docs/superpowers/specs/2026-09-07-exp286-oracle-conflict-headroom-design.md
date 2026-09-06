# EXP-286 Oracle Conflict-Core Headroom — Design

## 1. Goal and scientific boundary

Build a matched-neural **DEVELOPMENT** lane for frozen Stage-A `EXP-286` that measures the reasoning-cost headroom available from ground-truth conflict cores before investing in learned conflict localization.

This lane is strictly `EV-E2 / UNVERIFIED`.

It must not:

- consume confirmatory-open observations;
- derive or materialize post-freeze challenge randomness;
- execute the frozen scientific promotion/kill decision rule;
- relabel the existing `ConflictCoreRegion` as validated learned conflict localization;
- alter `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.

A positive DEVELOPMENT result only establishes that oracle conflict information is worth further confirmatory work. A negative result is allowed to simplify or remove downstream conflict machinery.

## 2. Frozen Stage-A authority

The implementation must preserve the exact `EXP-286` contract from `NLM-REASONING-STAGE-A-CONFIRMATORY-V1`:

- experiment: `EXP-286`;
- hypothesis: `H-CONFLICT-01`;
- question: **How much reasoning value comes from oracle unsat/minimal conflict cores?**
- arms:
  - `chronological_failure`: no conflict core; chronological rollback;
  - `oracle_conflict_core`: ground-truth conflict core supplied at contradiction;
- primary endpoint: `accounted_reasoning_flops_to_verified_solution`, direction `lower`;
- protected endpoint: `verified_solution_rate`, floor `oracle_conflict_core >= chronological_failure - 0.005`;
- MESI: relative reduction `0.15`;
- sample plan: paired, power target `0.90`, `min_n=32`, `max_n=128`, freeze from pilot log-cost paired variance with robust backup;
- analysis: paired log-cost ratio plus bootstrap CI; failures remain censored/scientific outcomes;
- multiplicity family: `CONFLICT_VALUE`;
- resource match:
  - oracle metadata is information, not free neural parameters;
  - oracle-information receipt must be reported;
  - both arms share the same maximum episode FLOP budget;
- challenge generation remains future-beacon-only after freeze;
- frozen decision rule is not executed in DEVELOPMENT.

## 3. Architectural choice

### Recommended and selected approach: experiment-local matched neural headroom court

Use the experiment-local matched-neural pattern established by EXP-277/EXP-279 rather than immediately wiring the full exact-16M candidate into a confirmatory-style executor.

Both arms use the same recurrent search/controller substrate, the same parameter envelope, the same paired conflict world, the same initialization lineage and the same maximum accounted-FLOP budget. The only experimental information difference is the contradiction-time conflict-core receipt.

This isolates the causal question EXP-286 is meant to answer: **does perfect conflict-core information create at least meaningful search-cost headroom at matched neural cost?**

### Rejected alternative A: train a learned conflict localizer first

This confounds two questions: whether conflict-core information is valuable and whether a learned locator can infer it. It risks spending complexity on a mechanism whose oracle upper bound may already be too small.

### Rejected alternative B: keep only the existing algorithmic proxy

The current Stage-A harness already compares chronological search with oracle conflict-variable priority under `EV-E2_ALGORITHMIC_PROXY`. That is useful smoke evidence but does not close the neural execution debt, neural parameter match, analytical FLOP accounting, artifact provenance or registry integration required for the next engineering layer.

## 4. Arm semantics and information separation

### 4.1 Shared substrate

Create two matched experiment-local neural arms with one identical module inventory and state initialization:

- surface/event encoder;
- variable-state encoder;
- recurrent deliberation/rollback controller;
- conflict-core adapter;
- rollback/priority head;
- verifier/solution head;
- exact target-parameter closure using existing budget utilities.

All optimizer-visible trainable parameters must be matched. No arm may gain hidden active capacity from an excluded reserve.

### 4.2 `chronological_failure`

The baseline must never receive the ground-truth conflict core.

At contradiction it receives a canonical **null-core receipt** plus chronological rollback context. The same conflict-core adapter is still executed on the null/chronology token so optimizer-visible parameter accounting remains matched, but the token contains no oracle variables, constraints, backjump target or future search information.

Its rollback order remains chronological.

### 4.3 `oracle_conflict_core`

The oracle arm receives a ground-truth minimal/local conflict-core artifact **only after the contradiction event is observed**.

The receipt may encode the variables/constraints implicated in the current contradiction, but it must not expose:

- future contradiction cores;
- the final solution;
- future action sequence;
- hidden challenge information;
- any oracle structure before contradiction time.

The oracle core is used to condition rollback/backjump priority within the same shared controller.

### 4.4 Required information receipt

The development artifact must record an exact receipt such as:

```json
{
  "artifact": "ground_truth_conflict_core",
  "ground_truth": true,
  "delivery_event": "after_current_contradiction_only",
  "delivered_to": ["oracle_conflict_core"],
  "withheld_from": ["chronological_failure"],
  "chronological_failure_received_conflict_core": false,
  "future_conflict_core_leakage": false,
  "solution_leakage": false
}
```

The semantic validator must reject any drift from this boundary even if the top-level artifact hash is recomputed.

## 5. Conflict-world generator

Create a deterministic neural development generator specialized for **globally solvable worlds with local contradictory branches**. This is preferred to a globally unsatisfiable-only world because the frozen primary metric is explicitly FLOPs to a **verified solution**.

Each paired world should contain:

- binary or small finite-domain variables;
- decoy branches that can consume chronological search work;
- at least one productive solution path;
- one or more local contradictory assignments with an exactly known minimal/local conflict core;
- deterministic contradiction-event metadata sufficient to generate the oracle receipt after the contradiction occurs;
- no arm-specific world mutation.

The generator must be seeded only with existing Stage-A lineage utilities:

```python
derive_stream_seed(root_seed, "EXP-286", replicate, rng_stream)
```

Allowed DEVELOPMENT streams:

- `augmentation` for training;
- `evaluation` for development evaluation;
- `model_init` for initial weights.

No confirmatory or challenge stream may be derived.

The same replicate must produce byte-identical paired tensors/metadata for both arms. Batch digests must bind tensor raw bytes plus lineage/geometry metadata.

## 6. Contradiction and core timing model

The generator/executor must model conflict-core delivery as an event, not as static privileged input.

For every episode:

1. both arms receive identical initial observations and surface state;
2. both execute the same budgeted search/controller loop;
3. when the current branch reaches a contradiction, the executor materializes the current contradiction event;
4. `chronological_failure` receives only the canonical null-core token;
5. `oracle_conflict_core` receives the current ground-truth minimal/local core;
6. each arm selects its rollback/priority action;
7. the loop continues until a verified solution is reached or the shared episode FLOP ceiling is exhausted.

This timing rule is a hard semantic boundary. A validator must reject artifacts indicating pre-contradiction oracle delivery.

## 7. Resource and parameter matching

The matched-pair audit must report at minimum:

- exact total parameters per arm;
- exact functional trainable parameters per arm;
- exact optimizer-visible parameters per arm;
- `parameter_match=true`;
- `functional_parameter_match=true`;
- `optimizer_visible_parameter_match=true`;
- no excluded reserve used to hide arm-specific functional capacity;
- identical initialization digest before training;
- shared declared maximum accounted FLOPs per episode;
- analytical compute ledger for shared encoder/controller, contradiction adapter, rollback head and verifier path;
- `hardware_profiler_flops_claimed=false`.

Oracle metadata is free **information** under the frozen contract, not free neural computation. Every neural operation that consumes that information is fully charged.

The chronological arm must execute a matched null-core adapter path so the oracle arm does not gain uncharged neural capacity simply because it has information to consume.

## 8. FLOP accounting and censored outcomes

The primary endpoint is exactly:

`accounted_reasoning_flops_to_verified_solution`

Lower is better.

For every episode, analytical FLOPs are accumulated along the actually executed path. The ledger must charge:

- encoding;
- recurrent deliberation/search steps;
- contradiction/core adapter invocation;
- rollback/priority scoring;
- verifier/solution checks;
- any additional oracle-conditioned neural step.

If an arm fails to reach a verified solution before the shared ceiling:

- the replicate is **not dropped**;
- `verified_solution_rate` records failure;
- cost is recorded at the shared censoring ceiling;
- `censored_at_max_flops=true` is emitted;
- the raw row remains part of development aggregates.

The development runner may report descriptive paired log-cost summaries, but it must not execute the frozen bootstrap-CI promotion/kill rule.

## 9. Training and evaluation

Use paired training/evaluation geometry with disjoint replicate lineages.

Training:

- RNG stream `augmentation`;
- same batch for both arms;
- same root/model-init lineage;
- same number of optimizer updates;
- same optimizer family and hyperparameters;
- no evaluation or confirmatory data mixed into training.

Development evaluation:

- RNG stream `evaluation`;
- replicate range disjoint from training;
- same paired world and contradiction sequence for both arms;
- raw per-replicate rows retained;
- no selective reruns for scientific failures.

## 10. Development artifact and validator

Create an append-only schema such as:

`NLM-EXP-286-PAIRED-DEV-EVAL-V1`

Required top-level scientific boundary fields:

- `evidence_level="EV-E2"`;
- `decision="UNVERIFIED"`;
- `confirmatory_ready=false`;
- `confirmatory_data_consumed=false`;
- `challenge_materialized=false`;
- `decision_rule_executed=false`;
- exact frozen protocol id/digest;
- source-tree/code digest;
- paired initial-state digest receipt;
- information-separation receipt;
- pair-audit plus digest;
- training lineage;
- evaluation raw rows;
- descriptive aggregate;
- explicit remaining confirmatory blockers;
- canonical self-hash.

The semantic validator must recompute or independently verify all security/scientific-critical values and reject at least:

- endpoint or MESI drift;
- protected-floor drift;
- arm ordering drift;
- protocol authority drift;
- pre-contradiction oracle-core delivery;
- baseline conflict-core receipt;
- future-core or solution leakage;
- initial-state mismatch;
- parameter/optimizer-visible mismatch;
- compute ceiling violation;
- missing/incorrect censoring on unresolved episodes;
- dropping or reordering failed replicates;
- training/evaluation lineage overlap;
- missing/duplicated batch digests;
- aggregate values inconsistent with raw rows;
- scientific promotion fields set true;
- artifact self-hash mismatch.

The validator must still reject semantic tampering after an attacker recomputes the top-level artifact digest.

## 11. Neural-arm registry integration

Extend `src/nolane_ai/experiments/neural_arm_registry.py` to include `EXP-286` in the neural target set and preserve exact frozen protocol arm descriptions.

Add optional evidence inputs:

- `exp286_pair_audit`;
- `exp286_execution_artifact`.

A valid matched-pair audit may clear implementation blockers for the two EXP-286 experiment-local neural arms. A valid paired DEVELOPMENT execution may advance a development status such as `PAIRED_CONFLICT_HEADROOM_DEV_READY`.

`match_court` must remain `BLOCKED` because confirmatory sample-size/analysis freeze, confirmatory-open execution and post-freeze challenge evidence remain outstanding.

The registry must not imply that the learned `ConflictCoreRegion.score_conflicts()` mechanism has been validated merely because oracle conflict information was useful.

## 12. Existing `ConflictCoreRegion` boundary

`ConflictCoreRegion` already provides a learned conflict scorer in the candidate model. EXP-286 DEVELOPMENT must treat that region as **implementation context**, not as a validated oracle-equivalent localizer.

The oracle lane answers an upper-bound/value question first. Learned localization becomes justified for a later lane only if oracle headroom is scientifically promising enough to warrant it.

Therefore this PR should not claim:

- learned core recovery accuracy;
- learned minimal-core fidelity;
- neural backjump superiority from learned localization;
- EV-E3 evidence for `ConflictCoreRegion`.

## 13. CLI, CI and version boundary

Add a CPU-safe development CLI, tentatively:

`scripts/run_exp286_paired_dev.py`

The CLI must:

1. refuse existing output paths before model execution;
2. verify the canonical frozen Stage-A protocol digest;
3. compute source-tree digest;
4. run tiny/default DEVELOPMENT geometry;
5. validate the execution artifact before writing;
6. optionally produce a neural-arm registry snapshot only from a valid artifact;
7. never derive confirmatory/challenge seeds.

CI model-smoke must include all EXP-286 unit/integration tests and one tiny paired development command while preserving EXP-277, EXP-279 and EXP-282 regressions.

After the full EXP-286 lane is green, bump the package version from `0.12.0` to `0.13.0` in both version authorities and document the DEVELOPMENT boundary in README.

## 14. Expected implementation files

Tentative production files:

- `src/nolane_ai/experiments/matched_conflict_arms.py`
- `src/nolane_ai/experiments/exp286_conflict_worlds.py`
- `src/nolane_ai/experiments/exp286_paired_runner.py`
- `scripts/run_exp286_paired_dev.py`
- modifications to `src/nolane_ai/experiments/neural_arm_registry.py`
- CI/package/README updates.

Tentative tests:

- `tests/test_exp286_neural_import.py`
- `tests/test_matched_conflict_arms.py`
- `tests/test_exp286_conflict_worlds.py`
- `tests/test_exp286_paired_runner.py`
- `tests/test_exp286_registry_integration.py`
- `tests/test_exp286_paired_cli.py`

## 15. Acceptance criteria

Implementation is complete only when all of the following are true:

1. a clean TDD RED run exists before EXP-286 production modules;
2. exact total/functional/optimizer-visible parameter matching passes;
3. the oracle-core receipt is contradiction-time-only and baseline leakage tests pass;
4. deterministic paired conflict-world regeneration tests pass;
5. actual-path analytical FLOP accounting and shared censoring-ceiling tests pass;
6. unresolved episodes remain in raw rows and aggregates;
7. semantic tamper tests fail closed even after re-hash;
8. registry evidence cannot clear confirmatory blockers or validate learned conflict localization;
9. tiny CLI execution passes in GitHub Actions;
10. all existing EXP-277, EXP-279 and EXP-282 regression/ceremony tests remain green;
11. frozen Stage-A protocol files are absent from the PR diff;
12. package/runtime versions agree at `0.13.0` only after the full lane is green;
13. exact-head core 3.11, core 3.13 and model-smoke CI are green;
14. reviewer pass has no unresolved Critical/Important issue;
15. squash merge uses exact expected-head protection and post-merge `main` CI is verified before engineering closure is claimed.
