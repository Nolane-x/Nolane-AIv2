# EXP-279 Neural Propagation/Branch/Hybrid Routing Design

**Status:** APPROVED FOR IMPLEMENTATION  
**Date:** 2026-09-06  
**Scope:** DEVELOPMENT-only neural execution lane for frozen `EXP-279`.

## 1. Purpose

Build an executable EV-E2 neural DEVELOPMENT lane for frozen EXP-279 that compares exactly three protocol arms:

- `propagation_only` — constraint propagation without branch search;
- `branch_only` — ARCS branch search without CBRF propagation;
- `hybrid` — propagation first, branch only when residual uncertainty remains.

The lane exists to close implementation, lineage, resource-accounting and evaluator provenance. It does **not** execute the frozen confirmatory decision rule and does **not** create EV-E3 evidence.

## 2. Frozen authority

`protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` remain byte-identical and authoritative.

EXP-279 frozen commitments copied into this design:

- primary endpoint: `verified_utility_per_accounted_flop_on_structure_dense_stratum`, higher is better;
- protected solution-rate floor: `hybrid >= best_simple - 0.01`;
- MESI: hybrid relative gain `0.08` over the best simpler arm;
- analysis: blocked paired contrasts, with Holm-adjusted comparisons against the best simpler arm at confirmatory analysis time;
- resource match: reclaimed parameters assigned to simpler rivals, equal maximum accounted FLOPs, predeclared structure-fit strata;
- paired sample plan: confirmatory minimum 32, maximum 128;
- challenge randomness: unavailable during development and derived only after future freeze/beacon ceremony.

Development artifacts must state:

- `evidence_level = EV-E2`;
- `decision = UNVERIFIED`;
- `confirmatory_ready = false`;
- `confirmatory_data_consumed = false`;
- `challenge_materialized = false`;
- `decision_rule_executed = false`.

## 3. Recommended architecture

Reuse the experiment-local neural substrate proven in EXP-277 rather than constructing full 16M regional arms immediately.

### 3.1 Shared neural substrate

All three arms share the same trainable module inventory and exact target parameter count:

- surface-event projection;
- variable-state projection;
- recurrent refinement core;
- propagation/message-passing transform;
- route/residual-uncertainty scorer;
- decision head;
- verifier-confidence head;
- capacity reserve for exact budget closure.

The three arms receive identical initial functional state. Arm behavior differs by the semantic role assigned to shared functional blocks and by whether hybrid routing conditionally executes its branch phase.

### 3.2 Arm semantics

`propagation_only`:

1. consumes surface events, variable states and compiled incidence;
2. performs propagation/message passing;
3. reclaims the recurrent functional block as iterative propagation refinement, never as branch search;
4. emits per-variable logits and verifier confidence.

`branch_only`:

1. consumes surface events and variable states;
2. **must not receive compiled incidence**;
3. reclaims the propagation transform as a branch preconditioner, then executes recurrent branch refinement;
4. emits per-variable logits and verifier confidence.

`hybrid`:

1. consumes surface events, variable states and compiled incidence;
2. performs propagation first;
3. computes an explicit residual-uncertainty score from propagation outputs;
4. executes branch refinement only for episodes whose residual uncertainty exceeds the predeclared route threshold;
5. emits route receipt, per-variable logits and verifier confidence.

The route threshold is an engineering configuration frozen in the development artifact before evaluation starts. Development results cannot be used to retroactively change a completed artifact.

## 4. Structure-fit strata

Use predeclared synthetic structure-dense strata, all generated from the same deterministic EXP-279 lineage machinery:

- `PROPAGATION_FIT` — each component is strongly anchored and propagation has direct structural support;
- `BRANCH_FIT` — surface evidence is sufficient while structural sharing provides less leverage, emphasizing recurrent branch refinement;
- `MIXED_RESIDUAL` — a mix of directly propagatable and residual-uncertainty components, intended to exercise conditional routing.

Each evaluation replicate records its stratum. Training may draw balanced augmentation batches across all three strata. Development evaluation uses the `evaluation` RNG stream and a deterministic blocked ordering with all three strata represented.

No stratum label is an oracle solution label; it describes generator geometry only. The incidence artifact remains truthful in every stratum.

## 5. Information separation

The same underlying world facts are serialized in two representations:

- surface/event representation available to every arm;
- compiled constraint-variable incidence available only to `propagation_only` and `hybrid`.

`branch_only` must have no forward argument or hidden path by which incidence is delivered. The paired artifact records an information receipt stating exactly which arms received incidence.

## 6. Active parameter reclaim and resource matching

All three arm objects have exact equal total and functional trainable parameter counts. The frozen phrase **“reclaimed parameters assigned to simpler rivals”** is interpreted strictly: simpler arms may not satisfy the match merely by carrying dormant functional modules.

Every non-reserve functional parameter must have an explicit active semantic assignment in each arm:

- `propagation_only` reuses the recurrent block as iterative propagation refinement and the route scorer as a propagation-confidence gate;
- `branch_only` reuses the propagation transform as a branch preconditioner and the route scorer as a branch-confidence gate;
- `hybrid` uses the propagation transform for propagation, the route scorer for residual uncertainty, and the recurrent block for conditional branch refinement.

The audit records:

- exact total parameter equality;
- exact functional parameter equality;
- exact **active functional parameter** equality;
- a per-arm reclaimed-parameter assignment receipt;
- each arm's executed-compute ledger;
- one shared declared `max_accounted_flops_per_episode` ceiling.

Capacity-reserve tensors exist only to close the exact parameter budget and are excluded from functional/active counts and optimizer groups. No functional parameter may be counted as reclaimed merely because it is present but unused.

Hardware profiler FLOPs are not claimed. The ledger is analytical scalar arithmetic accounting for the declared neural geometry. Conditional hybrid branch compute is charged according to the recorded route mask, while the shared resource ceiling is the maximum fully-routed episode cost.

## 7. Deterministic paired worlds

Create `Exp279RoutingGenerator` using only:

`derive_stream_seed(root_seed, "EXP-279", replicate, rng_stream)`.

Required streams:

- `model_init` for shared arm initialization;
- `augmentation` for development training worlds;
- `evaluation` for disjoint development evaluation worlds.

Every batch contains:

- `surface_events`;
- `variable_states`;
- `incidence`;
- `targets`;
- `stratum`;
- deterministic metadata and SHA-256 digest.

Training and evaluation replicate ranges must be disjoint. Re-running the same lineage and geometry must reproduce byte-identical tensors and digest.

## 8. Paired training and evaluation

Seed one shared arm state, clone it into all three arms, then train the three arms on the exact same ordered augmentation batches.

Training losses may differ because arm computation differs, but data lineage, optimizer configuration and initial functional state are matched.

For each evaluation replicate:

1. generate exactly one paired world batch;
2. run all three arms on that batch;
3. externally verify decisions against generator targets;
4. record verified solution rate, verified decision accuracy, verifier confidence, accounted FLOPs and normalized primary utility;
5. record the hybrid route fraction and route receipt;
6. retain the raw per-replicate row.

The development runner reports descriptive aggregate means and relative gains only. It must not run bootstrap/Holm confirmatory inference or emit a promotion/kill state.

## 9. Semantic validator

Artifact schema: `NLM-EXP-279-PAIRED-DEV-EVAL-V1`.

Validator must reject, even after an attacker recomputes the top-level digest:

- protocol ID/digest drift;
- arm ID/order drift;
- primary endpoint or MESI drift;
- protected floor drift;
- non-EV-E2 or non-UNVERIFIED claims;
- confirmatory/challenge materialization;
- train/eval overlap;
- missing/duplicated batch digests;
- missing structure-fit strata;
- branch-only incidence receipt;
- parameter or active-functional-parameter mismatch;
- missing reclaimed-parameter assignment;
- exceeded compute ceiling;
- route threshold/receipt inconsistency;
- aggregate values inconsistent with raw rows.

## 10. Neural-arm registry integration

Extend the existing Stage-A neural arm registry with optional:

- `exp279_pair_audit`;
- `exp279_execution_artifact`.

A valid triplet audit may clear implementation blockers for the three EXP-279 development arms. A valid paired execution may advance the development status to `PAIRED_ROUTING_DEV_READY`.

`match_court` remains `BLOCKED`. Remaining blockers must explicitly include confirmatory sample-size/analysis freeze, confirmatory-open execution and post-freeze challenge evidence.

No development artifact can promote EXP-279.

## 11. CLI and package boundary

Add `scripts/run_exp279_paired_dev.py`.

CLI requirements:

- verify canonical frozen Stage-A protocol digest before model execution;
- refuse to overwrite artifact or registry output;
- compute source-tree digest;
- support a CPU-safe `--tiny` lane;
- persist one append-only development artifact and registry snapshot;
- never derive confirmatory or challenge seeds.

Bump package to `0.12.0` only after the full EXP-279 development lane is green.

## 12. Acceptance criteria

Implementation is complete only when:

1. clean TDD RED evidence exists before production modules;
2. exact total/functional/**active-functional** parameter equality and shared-compute ceiling tests pass;
3. deterministic generator and strata tests pass twice;
4. paired runner and tamper validator tests pass;
5. neural registry integration tests pass;
6. CLI tiny run passes in GitHub Actions;
7. all existing EXP-277 and EXP-282 tests remain green;
8. frozen Stage-A protocol files are absent from the PR diff;
9. exact-head PR CI is fully green;
10. reviewer pass finds no unresolved Critical/Important issue before squash merge.
