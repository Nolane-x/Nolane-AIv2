# EXP-279 Neural Propagation/Branch/Hybrid Routing Design

## Status

Approved architectural design for the EXP-279 DEVELOPMENT execution lane. This document does not alter the frozen Stage-A V1 protocol and does not authorize confirmatory-open or post-freeze challenge execution.

## Scientific Authority

The authoritative frozen protocol remains `protocols/stage_a_v1.json` / `protocols/stage_a_v1.sha256`.

EXP-279 is frozen as:

- question: **Propagation-only vs branch-only vs hybrid: when does each win?**
- arms:
  - `propagation_only`: constraint propagation without branch search
  - `branch_only`: ARCS branch search without CBRF propagation
  - `hybrid`: propagation followed by branch search when residual uncertainty remains
- primary endpoint: `verified_utility_per_accounted_flop_on_structure_dense_stratum`, higher is better
- protected endpoint: `verified_solution_rate`, with `hybrid >= best_simple - 0.01`
- MESI: relative gain `0.08`
- sample-size contract: paired, confirmatory `min_n=32`, `max_n=128`, power target `0.90`
- analysis: blocked paired contrasts, Holm-adjusted comparisons against the best simpler arm
- multiplicity family: `PROPAGATION_ROUTING`
- resource match:
  - reclaimed parameters assigned to simpler rivals
  - equal max accounted FLOPs
  - predeclared structure-fit strata
- frozen decision rule: hybrid promotes only if it exceeds the best simpler arm by at least 8% with a noninferior solution rate; simpler-arm practical equivalence or superiority kills hybrid complexity.

This DEVELOPMENT lane must not execute that confirmatory decision rule.

## Epistemic Boundary

Every EXP-279 development artifact introduced by this design is:

- `EV-E2`
- `UNVERIFIED`
- `confirmatory_ready=false`
- `confirmatory_data_consumed=false`
- `challenge_materialized=false`
- `decision_rule_executed=false`

A successful development run proves execution/provenance/resource closure only. It does not prove the EXP-279 scientific hypothesis and cannot be reclassified as post-freeze challenge evidence.

## Design Choice

Reuse the matched-neural substrate established by EXP-277 rather than introducing a second unrelated neural stack.

### Shared arm substrate

All three arms use one experiment-local matched architecture with the same trainable parameter envelope and cloned functional initialization. The shared functional modules are:

- surface-event encoder
- variable-state encoder
- recurrent branch controller
- propagation transform
- decision head
- verifier-confidence head
- small deterministic hybrid routing gate

Capacity not used by a simpler arm remains trainable and optimizer-visible through an arm-local **reclaimed capacity pathway** rather than being hidden as inactive reserve. This is the concrete DEVELOPMENT implementation of the frozen rule that reclaimed parameters are assigned to simpler rivals.

The three arms therefore have exact equality for:

- total parameters
- functional trainable parameters
- optimizer-visible parameters
- initial functional-state digest

The differences are execution semantics, not parameter availability.

### Arm semantics

#### `propagation_only`

Receives `surface_events`, `variable_states`, and compiled `incidence` derived from the same paired world facts. It performs factor/variable propagation and does not execute the recurrent branch-search path. Its reclaimed branch-controller capacity is routed through the reclaimed-capacity pathway so the parameters remain active and trainable without granting branch search semantics.

#### `branch_only`

Receives `surface_events` and `variable_states` but **must not receive compiled incidence**. It executes recurrent branch-style deliberation. Its reclaimed propagation capacity is routed through the same shape-preserving reclaimed-capacity pathway.

This arm must fail closed if an `incidence=` argument is supplied.

#### `hybrid`

Receives the same surface representation and compiled incidence as propagation-only. It first performs propagation. A deterministic residual-uncertainty statistic then determines whether the recurrent branch path executes. The routing threshold is configuration frozen inside the DEVELOPMENT artifact before evaluation; it is not fitted from evaluation outcomes.

The hybrid artifact records, per replicate:

- residual uncertainty before routing
- whether branch path executed
- propagation accounted FLOPs
- branch accounted FLOPs actually charged
- total accounted FLOPs

No branch FLOPs may be omitted when the branch path executes.

## Structure-Dense Development Worlds

EXP-279 reuses the structure-dense world principle from EXP-277 but extends it with **predeclared structure-fit strata**. The generator remains deterministic under Stage-A RNG lineage and produces the same paired world for all three arms.

Development strata are fixed as:

1. `PROPAGATION_FRIENDLY`
   - relatively dense/local constraint connectivity
   - low ambiguity after propagation
   - designed to expose cases where branch overhead should be unnecessary

2. `BRANCH_FRIENDLY`
   - sparse/ambiguous constraint structure
   - propagation leaves materially unresolved assignments
   - designed to expose cases where branch deliberation should matter

3. `MIXED_STRUCTURE`
   - intermediate connectivity and ambiguity
   - designed to exercise the hybrid routing boundary

These labels describe DEVELOPMENT generator geometry, not empirical claims that an arm will win.

The generator must derive randomness only from:

`derive_stream_seed(root_seed, "EXP-279", replicate, rng_stream)`

Training uses `augmentation`; development evaluation uses `evaluation`. Training/evaluation replicate ranges must be disjoint. No `challenge` stream is derived or materialized.

Each batch includes:

- `surface_events`
- `variable_states`
- `incidence`
- `targets`
- `structure_fit_stratum`
- `residual_uncertainty_target` or equivalent generator truth used only for external validation/stratification, never given directly to a competing arm
- deterministic batch digest
- replicate and RNG-lineage metadata

## Oracle/Information Receipt

EXP-279 is not an oracle-headroom experiment, but compiled incidence is still privileged structural representation relative to branch-only. The artifact must explicitly bind this receipt:

- `propagation_only`: receives compiled incidence
- `hybrid`: receives compiled incidence
- `branch_only`: compiled incidence withheld

All three receive the same underlying paired world facts and target labels for training/evaluation supervision. The branch-only arm cannot infer a hidden direct pointer to the incidence tensor from metadata.

## Parameter/Reclaimed-Capacity Contract

The matched-arm audit must report for every arm:

- total parameters
- functional trainable parameters
- reclaimed-capacity parameters
- inactive/frozen reserve parameters, expected `0` for the experiment-local arm envelope
- optimizer-visible parameter count
- functional initialization digest

The audit fails if any arm differs on the matched counts or if a simpler arm silently parks reclaimed trainable capacity in an excluded reserve.

The artifact must label this as a DEVELOPMENT resource-matching construction, not proof that full 16M regional models are matched.

## Accounted Compute Contract

Use analytical scalar arithmetic accounting, explicitly labeled **not hardware-profiler FLOPs**.

For each geometry and arm, the ledger separately records:

- input/event encoding
- variable encoding
- propagation work
- recurrent branch work
- reclaimed-capacity work
- hybrid routing work
- decision/verifier heads
- total accounted FLOPs per episode

All arms share one declared `max_accounted_flops_per_episode`; execution fails if any arm exceeds it.

For hybrid, total charge is path-dependent. A replicate that routes to branch must include propagation + routing + branch + head costs. A replicate that stops after propagation is charged only the executed path plus shared components.

The primary development metric is computed only after charging the corresponding executed accounted cost:

`verified_utility_per_accounted_flop_on_structure_dense_stratum`

The metric name is preserved exactly from the frozen protocol even though the DEVELOPMENT run does not execute the frozen statistical decision rule.

## Paired Development Runner

The runner builds all three arms from one `model_init` seed and verifies identical initial functional digests before training.

Training:

- identical paired batches for all arms
- `augmentation` stream
- identical replicate order
- separate optimizers with identical optimizer hyperparameters
- no evaluation batches consumed during training

Evaluation:

- `evaluation` stream
- disjoint replicate indices
- same paired batch passed to all three arms
- external exact target verification
- per-replicate raw metrics preserved
- results grouped by predeclared stratum

Per arm/replicate record at minimum:

- `verified_solution_rate`
- `verified_decision_accuracy`
- `accounted_flops_per_episode`
- `verified_utility_per_accounted_flop_on_structure_dense_stratum`

Hybrid additionally records routing receipt and executed-path ledger.

Development aggregates may report descriptive pairwise relative gains and stratum means, but must not emit `PROMOTE_TO_NEXT_STAGE`, `KILL_SUBSYSTEM`, or any Holm-adjusted confirmatory conclusion.

## Semantic Validator

The artifact schema is:

`NLM-EXP-279-PAIRED-DEV-EVAL-V1`

The semantic validator must reject even correctly rehashed tampering of:

- schema/evidence/decision boundary
- protocol or code provenance presence
- arm IDs or order
- parameter/reclaimed-capacity matching
- branch-only incidence withholding
- shared-world pairing
- compute ceiling closure
- primary metric name/direction
- MESI `0.08`
- protected solution-rate floor `hybrid >= best_simple - 0.01`
- `PROPAGATION_ROUTING` multiplicity family
- the exact three predeclared structure-fit strata
- training/evaluation RNG streams or overlapping lineages
- evaluation replicate ordering/digests
- hybrid path receipt or path-dependent cost accounting
- aggregate values inconsistent with raw rows
- confirmatory/challenge flags
- top-level self-hash

The validator may validate descriptive development statistics but must not run confirmatory Holm inference.

## Neural Arm Registry Integration

`build_neural_arm_registry(...)` gains optional EXP-279 matched-audit and paired-execution evidence.

Valid matched-arm evidence upgrades the three EXP-279 implementation rows from current component/blocker states to experiment-local matched DEVELOPMENT implementations.

Valid paired execution changes EXP-279 development status to a value equivalent to:

`PAIRED_STRATIFIED_ROUTING_DEV_READY`

However:

- `match_court` remains `BLOCKED`
- confirmatory sample-size/blocked-analysis freeze remains open
- confirmatory-open execution remains unrun
- post-freeze challenge remains unrun

Supplying execution evidence without matched-audit evidence fails closed.

## CLI and Version Boundary

Add `scripts/run_exp279_paired_dev.py`.

The CLI must:

1. verify canonical frozen Stage-A V1 protocol digest before neural execution
2. refuse overwrite of requested output paths before training starts
3. bind source-tree digest
4. run only DEVELOPMENT RNG streams
5. optionally emit the updated neural-arm registry
6. write one append-only JSON development artifact
7. expose a CPU-safe `--tiny` mode for CI

No confirmatory or challenge command is introduced in this PR.

Package version moves from `0.11.0` to `0.12.0` only after the complete EXP-279 lane is green.

## CI / TDD Contract

Development follows RED -> GREEN with preserved GitHub Actions evidence.

Required tests cover:

- import contract
- exact matched parameters and reclaimed capacity
- arm input separation
- compute ledger and hybrid path charging
- deterministic stratified generator and lineage separation
- paired runner and semantic tamper rejection
- registry evidence gating
- CLI frozen-protocol/overwrite/no-confirmatory boundary
- all existing EXP-277 and EXP-282 regression suites

The explicit model-smoke job executes one tiny EXP-279 development run after unit tests.

## Non-Goals

This design does **not**:

- modify frozen Stage-A V1 protocol bytes
- claim EXP-279 is scientifically passed or failed
- perform confirmatory sample-size estimation/freeze
- perform Holm-adjusted confirmatory analysis
- consume confirmatory-open observations
- derive post-freeze challenge seeds
- replace full 16M regional model architecture
- claim analytical FLOP estimates are hardware profiler measurements

## Completion Criteria

The PR is complete only when:

- all three matched experiment-local neural arms are executable
- reclaimed-parameter matching is auditable and fail-closed
- structure-fit strata are predeclared and deterministic
- paired training/evaluation lineage is closed
- semantic validation rejects rehashed tampering
- registry integration remains scientifically blocked from confirmation
- tiny CLI execution is green in CI
- existing EXP-277/EXP-282 suites remain green
- frozen protocol files are absent from the PR diff
- exact-head PR CI is fully green before squash merge
