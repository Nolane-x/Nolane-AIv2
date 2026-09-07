# Nolane-AIv2 — NLM V0.16.1 100M Research Substrate

Nolane-AIv2 is the executable research repository for the **Nolane Living Model (NLM) V0.16.1** program. The project is intentionally built as a falsification lab: mechanisms earn complexity by surviving frozen Stage-A gates; structural correctness is never relabeled as neural capability evidence.

## Current authoritative boundary

- Candidate footprint: **exactly 100,000,000 parameters**.
- Frozen support allocation: **10,000,000**; trainable allocation: **90,000,000**.
- Canonical experiment namespace: `EXP-###`.
- Evidence maturity namespace: `EV-*`.
- Confirmatory Stage-A protocol remains frozen at `protocols/stage_a_v1.json`.
- Neural architecture claims remain **UNVERIFIED** until matched neural runs produce EV-E3 evidence.

## What is executable now

The repository now has two separate reasoning lanes.

### 1. Algorithmic Stage-A falsification lane

A dependency-light structured substrate implements:

- Canonical Problem State with finite-domain table constraints;
- generalized arc-consistency propagation;
- chronological branch search;
- propagation + branch hybrid search;
- oracle conflict-variable prioritization;
- episode-local nogood storage;
- explicit binary belief-state updates and a recurrent evidence baseline;
- bounded exact semantic-equivalence checking for compile-valid fidelity traps;
- paired open-seed proxy runners for `EXP-277`, `EXP-279`, `EXP-282`, `EXP-286`, `EXP-289`, and `EXP-297`, with every smoke arm separated from its `protocol_arm_id`;
- deterministic paired-effect/bootstrap summaries over an explicitly labeled abstract operation proxy;
- EV-E2 Evidence Packet generation locked to the frozen protocol digest and source-tree digest.

This lane measures whether cheap algorithmic proxies expose mechanism headroom. Every raw arm is marked `EV-E2_ALGORITHMIC_PROXY`; it is not a claim that the frozen protocol arm (for example full V0.15 ARCS) has been implemented. It cannot promote a neural claim by itself.

### 2. Exact-100M neural candidate lane

The 100M PyTorch candidate no longer routes every region through the same generic residual block. Four Stage-A-critical regions now expose specialized functional computation while preserving the exact V0.16 budget:

- `recurrent_deliberation_core` → recurrent GRU deliberation;
- `constraint_belief_fabric` → differentiable constraint↔variable message passing + belief logits;
- `conflict_core_backjump_clause` → learned conflict attribution scores;
- `problem_compiler_fidelity_court` → learned semantic-pair fidelity score.

The remaining unused budget is still represented as explicit `capacity_reserve`. This is deliberate: reserve is capacity, not evidence that a mechanism exists.

A model audit reports functional vs reserved parameters per region so architecture growth cannot hide behind the headline 100M count.

## Run the Stage-A engineering smoke lane

```bash
python -m pip install -e '.[dev]'
python scripts/verify_protocol.py
python scripts/run_stage_a.py --replicates 4 --output /tmp/stage-a-smoke.json
```

Run all 32 frozen open replicate indices:

```bash
python scripts/run_stage_a.py --full-open --output /tmp/stage-a-open-32.json
```

The runner always emits `EV-E2 / UNVERIFIED`. Its cost counters are explicitly labeled `ABSTRACT_OPERATIONS_NOT_HARDWARE_FLOPS`, so smoke effects cannot be interpreted as satisfying the confirmatory FLOP-based MESI. It is intentionally incapable of declaring the neural thesis verified.

## Audit the exact 100M candidate

```bash
python -m pip install -e '.[dev,model]'
python scripts/audit_model.py
```

The audit instantiates the authoritative candidate on PyTorch's `meta` device, so exact parameter accounting can be checked without allocating 100M real parameter values.

## Tests

```bash
pytest -q
python -m compileall -q src scripts
```

CI has a dependency-light core lane plus a separate PyTorch model lane. The model lane verifies exact 100M accounting, tiny forward execution, structured reasoning shapes, Stage-A loss backpropagation and reserve isolation.

## Evidence discipline

- A protocol hash establishes committed bytes, not scientific truth.
- Open-smoke results are development/process evidence, not EV-E3 neural evidence.
- Architecture-caused divergence remains a scientific outcome.
- Practical equivalence selects the simpler rival.
- Post-freeze challenge randomness remains unavailable until code/config/evaluator/analysis freeze.
- A negative gate is allowed to delete a subsystem.
- `capacity_reserve` must not be counted as implemented capability.

## Stage-A neural pilot (16M wind tunnel)

The next execution layer is an exact **16,000,000 parameter** neural pilot for the first six Stage-A gates. It deliberately keeps only problem binding, recurrent deliberation, constraint-belief reasoning, conflict scoring, semantic fidelity, verifier support and explicit reserve capacity.

Audit the real 16M parameter topology without allocating storage:

```bash
python scripts/audit_stage_a_pilot.py
```

Run a tiny CPU training smoke that exercises the same optimizer/curriculum/checkpoint contracts:

```bash
python scripts/train_stage_a_pilot.py \
  --tiny \
  --steps 2 \
  --batch-size 4 \
  --variables 4 \
  --constraints 3 \
  --output-dir /tmp/nlm-stage-a-pilot
```

The trainer excludes every `capacity_reserve` tensor from optimizer groups. Checkpoint manifests bind protocol, source tree, model config, curriculum, RNG lineage and tensor-file SHA-256. Training-smoke checkpoints are structurally capped at `EV-E2 / UNVERIFIED`; they cannot self-promote to neural evidence.

The default non-`--tiny` runner instantiates the exact 16M pilot on CPU. This is an execution substrate, not a capability claim or a substitute for frozen EV-E3 experiments.

## EXP-277 matched neural DEVELOPMENT lane

Package `0.11.0` adds an executable matched-neural development lane for the frozen `EXP-277` oracle-structure headroom gate. It intentionally remains **EV-E2 / UNVERIFIED**: the lane measures development behavior and closes execution provenance, but it does not consume confirmatory-open data or execute the frozen scientific promotion rule.

Run the CPU-safe development smoke:

```bash
python scripts/run_exp277_paired_dev.py \
  --tiny \
  --train-replicates 2 \
  --eval-replicates 2 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 4 \
  --constraints 2 \
  --output /tmp/nlm-exp277-paired.json \
  --registry-output /tmp/nlm-exp277-registry.json
```

The two arms share functional initialization and exact parameter count. `arcs_branch` receives only the surface event/variable representation; `oracle_cbrf` additionally receives the ground-truth constraint↔variable incidence artifact. Both are charged by an explicit analytical accounted-FLOP ledger under one declared per-episode ceiling and run on the exact same paired world lineage. The CLI requires the canonical frozen Stage-A protocol digest and refuses to overwrite either output before any experiment execution.

The resulting registry state is `PAIRED_STRUCTURE_DENSE_DEV_READY` with `match_court=BLOCKED`. Remaining confirmatory sample-size/analysis freeze, confirmatory-open execution, and post-freeze challenge evidence are deliberately not manufactured by this development lane.

## EXP-279 matched neural routing DEVELOPMENT lane

Package `0.12.0` adds a three-arm neural development lane for the frozen `EXP-279` propagation-routing gate. It compares `propagation_only`, `branch_only`, and `hybrid` on paired deterministic worlds spanning the predeclared `PROPAGATION_FIT`, `BRANCH_FIT`, and `MIXED_RESIDUAL` structure-fit strata.

Run the CPU-safe development smoke:

```bash
python scripts/run_exp279_paired_dev.py \
  --tiny \
  --train-replicates 3 \
  --eval-replicates 3 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 4 \
  --constraints 2 \
  --route-threshold 0.5 \
  --output /tmp/nlm-exp279-paired.json \
  --registry-output /tmp/nlm-exp279-registry.json
```

All three arms share one matched parameter envelope. `branch_only` is structurally denied the compiled constraint↔variable incidence artifact, while `propagation_only` and `hybrid` receive it. Simpler arms reclaim otherwise idle functional capacity rather than hiding it in excluded reserve. The hybrid executes branch recurrence only for episodes whose residual uncertainty crosses the frozen development threshold, and its cost receipt charges the observed routed fraction between pre-accounted stop and branch paths under one common maximum FLOP ceiling.

This lane remains **EV-E2 / UNVERIFIED**. It reports descriptive paired development aggregates only; it does not run the frozen blocked/Holm confirmatory analysis, does not consume confirmatory-open observations, and does not materialize post-freeze challenge randomness. The Neural Arm Registry can record `PAIRED_ROUTING_DEV_READY`, but `match_court` remains `BLOCKED` until confirmatory sample-size/analysis freeze, confirmatory-open execution, and post-freeze challenge evidence are actually completed.

## EXP-286 oracle conflict-core DEVELOPMENT lane

Package `0.13.0` adds a matched-neural DEVELOPMENT lane for the frozen `EXP-286` oracle conflict-core headroom gate. It compares the exact frozen arms `chronological_failure` and `oracle_conflict_core` on deterministic, globally solvable conflict worlds with paired model/world initialization and one common analytical accounted-FLOP ceiling.

Run the CPU-safe development smoke:

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
  --registry-output /tmp/nlm-exp286-registry.json
```

`chronological_failure` never receives a ground-truth conflict core; it executes the matched null-core path plus chronological rollback context. `oracle_conflict_core` receives the current ground-truth local/minimal conflict core only after the current contradiction event. Future cores, solution targets as privileged inputs, confirmatory observations, and challenge randomness are withheld. The emitted artifact explicitly records `challenge_seed_materialized=false`, `challenge_materialized=false`, `confirmatory_data_consumed=false`, and `decision_rule_executed=false`.

Both arms expose exact total/functional/optimizer-visible parameter matching, identical functional initialization, deterministic paired lineage, and an analytical compute ledger. Neural operations that consume oracle metadata are charged. Unresolved episodes are retained as scientific outcomes and censored at the shared FLOP ceiling rather than dropped. The development runner reports descriptive paired log-cost/headroom statistics only; it does **not** execute the frozen confirmatory bootstrap decision rule.

A valid execution can advance the Neural Arm Registry to `PAIRED_CONFLICT_HEADROOM_DEV_READY`, but `match_court` remains `BLOCKED`. This lane remains **EV-E2 / UNVERIFIED** and does not validate the learned `ConflictCoreRegion` localizer. Confirmatory sample-size/analysis freeze, confirmatory-open execution, and post-freeze challenge evidence remain separate future gates.

## EXP-289 episode-local nogood DEVELOPMENT lane

Package `0.14.0` adds the matched-neural DEVELOPMENT lane for frozen `EXP-289`. The exact frozen arms are `no_nogood` (hybrid reasoner without episode-local learned nogoods) and `local_nogood` (the same reasoner with an episode-local nogood store and scoped applicability). Both arms share the same trainable neural envelope, initialization lineage, deterministic world/restart lineage, and common maximum accounted-cost ceiling.

Run the CPU-safe development smoke:

```bash
python scripts/run_exp289_paired_dev.py \
  --tiny \
  --train-replicates 2 \
  --eval-replicates 3 \
  --batch-size 2 \
  --timesteps 3 \
  --restarts 3 \
  --variables 6 \
  --decoys 2 \
  --max-search-steps 12 \
  --output /tmp/nlm-exp289-paired.json \
  --registry-output /tmp/nlm-exp289-registry.json
```

`local_nogood` may store only exact, non-empty partial assignments that its own current-episode search actually reached and observed as dead ends. Reuse is exact subset matching within the same episode/problem only: there is no cross-episode or cross-problem import, approximate retrieval, oracle conflict core, learned clause transfer, or lifelong lemma mechanism. The evaluator's valid-completion truth is used only after an arm action to audit soundness; it never gates insertion, chooses a controller action, or enters an arm input.

The frozen primary endpoint is `repeat_dead_end_rate`, lower is better, with a 25% relative-reduction MESI. Its denominator is arm-independent and predeclared by the deterministic generator/evaluator, so an arm cannot improve the metric by changing which repeat opportunities count. Zero-opportunity episodes remain in raw evidence under the explicit exclusion policy. Protected safety evidence includes `valid_state_overprune_rate <= 0.005` and the frozen verified-solution-rate floor `local_nogood >= no_nogood - 0.01`; truthful over-prune or ceiling-exhaustion outcomes remain scientific failures rather than being deleted or repaired.

Memory is not free compute. Canonicalization, insertion, query and exact subset-comparison operations are charged alongside the matched neural compute path, and the validator reconstructs raw cost/censoring receipts under one common ceiling. The CLI verifies the canonical frozen protocol digest before model work, refuses overwrite, validates execution/registry semantics before publication, and writes fully staged JSON artifacts without exposing partial files.

This lane remains **EV-E2 / UNVERIFIED**. It emits `confirmatory_ready=false`, `confirmatory_data_consumed=false`, `challenge_seed_materialized=false`, `challenge_materialized=false`, and `decision_rule_executed=false`. A valid DEVELOPMENT execution can advance the Neural Arm Registry to `PAIRED_LOCAL_NOGOOD_DEV_READY`, but `match_court` remains `BLOCKED`. EXP-289 does **not** validate EXP-290-style learned cross-problem clause transfer and does **not** establish lifelong lemma economy; those are separate future claims and gates.

## EXP-297 Encoding Fidelity Court DEVELOPMENT lane

Package `0.15.0` adds the matched-neural DEVELOPMENT lane for the final frozen Stage-A engineering gate, `EXP-297`. The frozen arms remain `compile_only` and `fidelity_court`, but both execute the same neural comparison substrate: source/candidate encoding, recurrent comparison, `FidelityCourtRegion`, receipt adapter, authority head and verifier head. `compile_only` receives a canonical null-fidelity receipt so it cannot gain a compute advantage by skipping the neural fidelity path.

The deterministic evaluator generates 16 compile-valid candidates per replicate across eight semantic strata: `faithful_equivalent`, `relation_shift`, `constraint_drop`, `constraint_strengthen`, `constraint_weaken`, `variable_binding_swap`, `domain_mapping_error`, and `negation_or_relation_flip`. Each stratum contains both faithful and wrong candidates. Construction provenance supplies evaluator truth, while arm-observable inputs exclude the faithful/wrong label and trap stratum.

The semantic court is fail-closed. It searches for independent bidirectional witnesses: an assignment accepted by the source and rejected by the candidate, or an assignment accepted by the candidate and rejected by the source. A verified divergence returns `court_reject`; exact closure without a witness returns `court_accept`; exceeding the bounded exact-assignment ceiling returns `court_inconclusive`, which never grants semantic authority.

Run the CPU-safe development smoke:

```bash
python scripts/run_exp297_paired_dev.py \
  --tiny \
  --eval-replicates 2 \
  --eval-start-replicate 1000 \
  --output /tmp/nlm-exp297-paired.json \
  --registry-output /tmp/nlm-exp297-registry.json
```

The artifact schema is `NLM-EXP-297-PAIRED-DEV-EVAL-V1`. Its reconstruction validator regenerates candidate lineage and recomputes compile results, witnesses, authority decisions, semantic verification operations, confusion matrices, balanced accuracy and both protected endpoints from raw per-candidate evidence. Neural accounted FLOPs and symbolic semantic-verification operations are reported separately; neither is mislabeled as hardware-profiler FLOPs.

This lane remains strictly **EV-E2 / UNVERIFIED**. Development outputs keep `confirmatory_ready=false`, `confirmatory_data_consumed=false`, `challenge_seed_materialized=false`, `challenge_materialized=false`, `decision_rule_executed=false`, `hidden_trap_family_consumed=false`, and `semantic_authority_promoted=false`. A valid development execution may record `PAIRED_FIDELITY_COURT_DEV_READY`, but `match_court` remains `BLOCKED`. Descriptive development balanced accuracy—even if numerically high—does not satisfy the frozen `+0.10` confirmatory MESI and does not production-validate semantic authority.
