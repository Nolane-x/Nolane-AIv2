# EXP-289 Episode-Local Nogood Learning — Design

## 1. Goal and scientific boundary

Build a matched-neural **DEVELOPMENT** lane for frozen Stage-A `EXP-289` that asks whether episode-local nogoods reduce logically repeated dead ends enough to justify their maintenance cost without over-pruning valid states.

This lane is strictly `EV-E2 / UNVERIFIED`.

It must not:

- consume confirmatory-open observations;
- derive or materialize post-freeze challenge randomness;
- execute the frozen scientific promotion/kill rule;
- claim cross-episode learned-clause transfer;
- claim lifelong lemma-economy value;
- claim learned conflict localization or learned clause generalization;
- alter `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.

A positive DEVELOPMENT result only establishes engineering headroom for exact, scoped, episode-local nogood reuse. A negative result is allowed to simplify or remove downstream clause-learning machinery.

## 2. Frozen Stage-A authority

The implementation must preserve the exact `EXP-289` contract from `NLM-REASONING-STAGE-A-CONFIRMATORY-V1`:

- experiment: `EXP-289`;
- hypothesis: `H-CONFLICT-01`;
- question: **Do episode-local nogoods reduce repeat dead ends without over-pruning?**
- arms:
  - `no_nogood`: hybrid reasoner without episode-local learned nogoods;
  - `local_nogood`: same reasoner with episode-local nogood store and scoped applicability;
- primary endpoint: `repeat_dead_end_rate`, direction `lower`;
- protected endpoints:
  - `valid_state_overprune_rate <= 0.005`;
  - `verified_solution_rate`, floor `local_nogood >= no_nogood - 0.01`;
- MESI: relative reduction `0.25`;
- sample plan: paired, power target `0.90`, `min_n=32`, `max_n=128`, freeze from pilot paired RDER variance with beta-binomial/bootstrap backup if bounded-rate assumptions fail;
- analysis: paired relative RDER reduction with bootstrap CI and explicit over-prune safety bound;
- multiplicity family: `LOCAL_NOGOOD`;
- resource match:
  - matched parameter budget;
  - nogood storage and retrieval charged to accounted cost;
  - matched episode budget;
- challenge generation remains future-beacon-only after freeze;
- frozen decision rule is not executed in DEVELOPMENT.

## 3. Architectural choice

### Recommended and selected approach: matched neural hybrid reasoner plus exact episode-scoped symbolic sidecar

Use an experiment-local matched neural hybrid reasoner for both arms. The arms share identical neural modules, initialization lineage, optimizer-visible parameters, restart schedule, world lineage and maximum episode budget.

The only causal difference is whether exact dead-end states observed during the current episode may be retained in and queried from an **episode-local exact nogood sidecar**.

`local_nogood` may store only dead-end partial assignments that the reasoner has actually reached and that the executor can independently certify inconsistent under the current world. `no_nogood` executes the same memory-adapter/null path and the same accounting hooks but never persists or applies a nogood.

This isolates the frozen question: **does exact local failure reuse reduce repeated dead-end work enough to justify its memory/lookup cost without pruning valid solution states?**

### Rejected alternative A: direct reuse of EXP-279 routing implementation as the scientific arm

EXP-279 established a matched propagation/branch/hybrid routing court, but binding EXP-289 directly to that experiment's implementation would couple two frozen hypotheses and make future changes to one experiment alter the other's provenance. EXP-289 should reuse low-level patterns/utilities, not scientific artifact authority.

### Rejected alternative B: neural associative nogood memory

A learned memory would confound three questions: whether nogoods are useful, whether retrieval can learn structural equivalence, and whether approximate retrieval is safe. Those belong to later learned-clause/generalization experiments, especially EXP-290.

### Rejected alternative C: keep only the existing algorithmic proxy

The current Stage-A harness uses `EpisodeNogoodStore` with repeated search calls. That is valuable `EV-E2_ALGORITHMIC_PROXY` smoke evidence but does not close matched neural parameters, deterministic restart opportunities, maintenance-cost accounting, exact over-prune verification, artifact provenance or registry integration.

## 4. Arm semantics

### 4.1 Shared neural substrate

Create two experiment-local neural arms with an identical module inventory:

- event/problem encoder;
- variable-state encoder;
- propagation/deliberation controller;
- branch/priority head;
- nogood-query adapter;
- verifier/solution head;
- exact target-parameter closure using existing budget utilities.

Both arms execute the nogood-query adapter each search step. The baseline receives a canonical null memory-response token, preventing hidden capacity or optimizer-visible parameter asymmetry.

### 4.2 `no_nogood`

The baseline:

- never persists dead-end assignments;
- never prunes because of a stored nogood;
- receives only canonical null memory-response tokens;
- follows the same bounded restart/search protocol as the local arm;
- pays the shared neural adapter cost but no real store insertion/match cost beyond a declared null-query accounting floor if needed for symmetric executor control flow.

### 4.3 `local_nogood`

The treatment arm owns an exact store scoped to one episode.

It may store a nogood only when:

1. the current partial assignment has been reached through the arm's actual search path;
2. the executor has observed a genuine dead end under the current world;
3. the partial assignment is non-empty;
4. the assignment is canonicalized deterministically;
5. the store scope exactly matches the current `episode_digest` and `problem_digest`.

A stored nogood is an exact conjunction of variable assignments. It matches a later partial state only when the stored assignment is a subset of the current assignment.

No approximate similarity retrieval is allowed in EXP-289.

## 5. Episode scope and leakage boundary

Every store must bind to an exact scope receipt:

```json
{
  "scope": "episode_local",
  "episode_digest": "...",
  "problem_digest": "...",
  "cross_episode_reuse": false,
  "cross_problem_reuse": false,
  "oracle_conflict_core_used": false,
  "learned_clause_generalization_claimed": false
}
```

The store is created at episode start and destroyed at episode end.

The semantic validator must reject:

- reuse of a nogood from another episode;
- reuse across problem/world digests;
- empty/root nogoods;
- insertion before an observed dead end;
- oracle conflict-core or future-path information entering the stored key;
- any claim that exact episode-local reuse demonstrates structural transfer.

## 6. Soundness rule for stored nogoods

EXP-289 must separate **search usefulness** from **logical safety**.

For a canonical partial assignment `N`, the executor may admit it to the store only if the current world independently verifies that no valid complete assignment extends `N`.

For bounded DEVELOPMENT worlds, the evaluator must be able to enumerate or otherwise deterministically verify the valid completion set.

The artifact must retain a store receipt for every insertion:

- canonical nogood key;
- insertion step/restart index;
- dead-end evidence digest;
- scope digests;
- independent `no_valid_completion=true` result;
- charged insertion/canonicalization cost.

The validator must independently regenerate the world and re-check this property rather than trust the stored boolean.

## 7. Development world and restart generator

Create deterministic globally solvable worlds designed to expose **repeat opportunities** across bounded restarts without giving either arm privileged knowledge.

Each episode should include:

- a finite-domain CSP/search world with at least one verified solution;
- multiple decoy regions capable of creating logically identical dead-end partial assignments under different search orders;
- a predeclared bounded restart schedule;
- deterministic variable/domain order perturbations per restart;
- evaluator-only canonical dead-end equivalence metadata;
- exact valid-solution enumeration or bounded verification support;
- no arm-specific world mutation.

Seed derivation must use:

```python
derive_stream_seed(root_seed, "EXP-289", replicate, rng_stream)
```

Allowed DEVELOPMENT streams:

- `model_init` for shared model initialization;
- `augmentation` for training/development fitting;
- `evaluation` for paired development evaluation;
- optionally `controller_noise` only if a predeclared neural controller-noise path is actually required and applied identically to both arms.

No confirmatory or challenge randomness may be derived.

The same replicate must regenerate byte-identical world tensors, restart schedule and evaluator opportunity metadata for both arms.

## 8. Repeat-dead-end opportunity contract

The primary endpoint must not use a denominator that an arm can manipulate by changing its own path.

For each paired episode, the generator/evaluator freezes a set of canonical **repeat opportunities** before arm execution. An opportunity represents a logically equivalent dead-end state that can be encountered in a later restart under the shared restart schedule.

The DEVELOPMENT primary endpoint is:

```text
repeat_dead_end_rate = repeated_dead_end_reentries / predeclared_repeat_opportunities
```

where:

- `predeclared_repeat_opportunities` is arm-independent and regenerated from the paired world/restart schedule;
- `repeated_dead_end_reentries` counts actual later re-entry into a canonical dead-end equivalence class already established earlier in the episode;
- preventing a repeat by a sound nogood reduces the numerator but cannot shrink the denominator;
- if the evaluator finds zero repeat opportunities, the episode remains in raw evidence but is excluded from the RDER denominator under one explicit predeclared rule reported in the artifact; it must not be silently dropped.

The artifact must also report descriptively:

- `prevented_repeat_count`;
- `nogood_hits`;
- `unique_dead_end_count`;
- raw dead-end signatures;
- per-restart dead-end lineage;
- opportunity count.

The frozen confirmatory bootstrap/MESI rule is not executed in DEVELOPMENT.

## 9. Over-prune safety court

`valid_state_overprune_rate` must be independently auditable.

For every stored nogood, the evaluator checks it against all bounded valid solutions/valid prefixes available from the generated world.

A safety violation occurs if a stored nogood would match any state that lies on at least one valid complete solution path.

Development reporting must include:

```text
valid_state_overprune_rate = invalidly_pruned_valid_states / evaluated_valid_states
```

plus raw offending state/nogood digests when nonzero.

The validator must recompute this from regenerated world truth and raw store receipts. It must reject an artifact that edits only the aggregate rate or deletes offending rows after re-hashing.

The frozen protected floor remains `<=0.005`; DEVELOPMENT records the metric but does not execute the confirmatory promotion/kill rule.

## 10. Search, restart and causal-divergence semantics

Pairing freezes:

- initial world;
- model initialization;
- restart schedule;
- exogenous RNG lineage;
- maximum episode budget;
- evaluator repeat-opportunity set.

Pairing does **not** require the arms to visit the same later states. Avoiding a dead end because of a sound stored nogood is the intended causal effect.

For each restart:

1. both arms begin from the same restart initial state/order;
2. each executes its own search path;
3. observed dead ends are independently verified;
4. baseline records but does not persist them;
5. local arm may insert sound non-empty nogoods;
6. later local queries may prune states matched by the exact current-episode store;
7. search continues until verified solution or the shared episode ceiling is exhausted.

Scientific/search failures are retained. Selective reruns are forbidden except predeclared infrastructure failures.

## 11. Parameter and resource matching

The matched-pair audit must report at minimum:

- exact total parameters per arm;
- exact functional trainable parameters per arm;
- exact active functional parameters per arm;
- exact optimizer-visible parameters per arm;
- identical functional initialization digest;
- same neural module inventory;
- same restart ceiling and maximum episode budget;
- exact store/query accounting policy;
- `parameter_match=true`;
- `functional_parameter_match=true`;
- `active_functional_parameter_match=true`;
- `optimizer_visible_parameter_match=true`;
- `memory_scope_closed=true`;
- `compute_budget_closed=true`;
- `hardware_profiler_flops_claimed=false`.

No excluded reserve may hide arm-specific active neural capacity.

The exact symbolic store is experimental information/state, not free neural parameters. Its storage, canonicalization and lookup work must be charged explicitly.

## 12. Accounted-cost ledger

EXP-289's primary scientific metric is RDER, but resource matching still requires a deterministic accounted-cost court.

The analytical ledger must separate at least:

- shared neural encoding FLOPs;
- propagation/deliberation FLOPs;
- branch/priority FLOPs;
- verifier FLOPs;
- nogood-adapter neural FLOPs;
- nogood canonicalization operations;
- store insertion operations;
- exact subset-match comparisons/retrieval operations;
- total `accounted_reasoning_cost`;
- declared maximum accounted cost per episode.

All actual path-dependent costs must be accumulated from raw step receipts.

No hardware-profiler FLOP claim is permitted.

If an episode exhausts the common budget:

- it is retained as a scientific outcome;
- solution failure is recorded;
- the cost is censored at the common ceiling;
- raw restart/search/store receipts remain present.

## 13. Training and evaluation

Training and evaluation use disjoint replicate lineages.

Training:

- `augmentation` stream;
- identical paired initialization;
- same optimizer family/hyperparameters/update count;
- no evaluation/confirmatory data mixed into training;
- exact store semantics remain episode-local during training;
- no cross-episode learned memory is introduced.

Development evaluation:

- `evaluation` stream;
- replicate range disjoint from training;
- shared paired worlds/restart schedules/opportunity sets;
- causal divergence retained and charged;
- all raw outcomes retained;
- no selective reruns for scientific failure.

## 14. Development artifact and semantic validator

Create an append-only schema such as:

`NLM-EXP-289-PAIRED-DEV-EVAL-V1`

Required top-level fields include:

- `evidence_level="EV-E2"`;
- `decision="UNVERIFIED"`;
- `confirmatory_ready=false`;
- `confirmatory_data_consumed=false`;
- `challenge_seed_materialized=false`;
- `challenge_materialized=false`;
- `decision_rule_executed=false`;
- exact protocol id/digest;
- source-tree/code digest;
- model-init and paired initial-state receipt;
- exact scope-policy receipt;
- pair audit plus digest;
- training lineage;
- evaluation raw rows;
- raw per-restart/per-step dead-end/store/query receipts;
- evaluator opportunity receipts;
- descriptive aggregates;
- explicit remaining blockers;
- canonical artifact self-hash.

The validator must independently regenerate or recompute all critical semantics and reject at least:

- arm/protocol/endpoint/MESI/protected-floor drift;
- initial-state or parameter mismatch;
- training/evaluation lineage overlap;
- episode/problem scope drift;
- cross-episode store reuse;
- empty/root nogoods;
- store insertion without an observed independently verified dead end;
- a nogood with at least one valid completion;
- approximate/fuzzy matching masquerading as exact local retrieval;
- repeat-opportunity denominator drift;
- repeated-dead-end numerator drift;
- deleted or reordered dead-end/restart rows;
- forged `prevented_repeat_count` or `nogood_hits`;
- over-prune aggregate drift;
- candidate/verified solution drift;
- storage/retrieval cost omission;
- per-step or episode accounted-cost drift;
- ceiling/censoring drift;
- cross-episode generalization claims;
- confirmatory/challenge/promotion fields set true;
- remaining-blocker removal;
- artifact digest mismatch.

The validator must continue to reject semantic tampering after the attacker recomputes the top-level self-hash.

## 15. Neural-arm registry integration

Extend `src/nolane_ai/experiments/neural_arm_registry.py` to include `EXP-289` while preserving exact frozen arm descriptions.

Add optional evidence inputs:

- `exp289_pair_audit`;
- `exp289_execution_artifact`.

A valid pair audit plus paired DEVELOPMENT artifact may advance an engineering status such as `PAIRED_LOCAL_NOGOOD_DEV_READY`.

`match_court` must remain `BLOCKED` because confirmatory sample-size/analysis freeze, confirmatory-open execution and post-freeze challenge evidence remain outstanding.

Registry language must not imply:

- EXP-290 learned-clause transfer is validated;
- lifelong lemma reuse is validated;
- conflict-core localization is validated;
- confirmatory H-CONFLICT-01 evidence exists.

## 16. Relationship to existing `EpisodeNogoodStore`

`src/nolane_ai/reasoning/search.py` already contains `EpisodeNogoodStore` and algorithmic branch/hybrid hooks.

EXP-289 may reuse its exact subset-match semantics or refactor common exact-store primitives if needed, but the experiment-specific development store must add stronger provenance requirements:

- explicit episode/problem scope;
- insertion evidence;
- deterministic canonical serialization;
- maintenance-cost receipts;
- safe reset/destruction boundary;
- validator-reconstructable lineage.

No unrelated rewrite of general search is required.

## 17. CLI, CI and version boundary

Add a CPU-safe development CLI, tentatively:

`scripts/run_exp289_paired_dev.py`

The CLI must:

1. refuse existing output paths before execution;
2. verify the canonical frozen Stage-A protocol digest;
3. compute source-tree digest;
4. run tiny/default DEVELOPMENT geometry;
5. explicitly validate the execution artifact before writing;
6. optionally generate a neural-arm registry snapshot only from validated evidence;
7. never derive confirmatory/challenge randomness.

CI model-smoke must include all EXP-289 unit/integration tests and one tiny paired DEVELOPMENT command while preserving EXP-277/279/282/286 regressions.

Only after the complete EXP-289 lane is green should package/runtime version advance from `0.13.0` to `0.14.0`, with README documenting the `EV-E2 / UNVERIFIED` boundary.

## 18. Tentative implementation files

Production:

- `src/nolane_ai/experiments/matched_nogood_arms.py`
- `src/nolane_ai/experiments/exp289_nogood_worlds.py`
- `src/nolane_ai/experiments/exp289_paired_runner.py`
- `scripts/run_exp289_paired_dev.py`
- modifications to `src/nolane_ai/experiments/neural_arm_registry.py`
- CI/package/README updates;
- only narrowly scoped common-search/store refactoring if required by exact provenance/accounting.

Tests:

- `tests/test_exp289_neural_import.py`
- `tests/test_matched_nogood_arms.py`
- `tests/test_exp289_nogood_worlds.py`
- `tests/test_exp289_paired_runner.py`
- `tests/test_exp289_registry_integration.py`
- `tests/test_exp289_paired_cli.py`

## 19. Acceptance criteria

Implementation is complete only when all of the following are true:

1. a clean TDD RED exists before EXP-289 production modules;
2. exact total/functional/active/optimizer-visible parameter matching passes;
3. both arms share identical initialization/world/restart/opportunity lineage;
4. local store is strictly episode/problem scoped and reset at episode end;
5. only observed, independently unsatisfiable, non-empty partial assignments can be stored;
6. baseline cannot persist/apply a real nogood;
7. repeat-opportunity denominator is arm-independent and validator-regenerated;
8. repeat re-entry/prevention receipts are reconstructable from raw restart/search traces;
9. valid-state over-prune safety is independently recomputed from regenerated ground truth;
10. storage/canonicalization/retrieval costs are charged and cannot be edited away;
11. unresolved/censored scientific outcomes remain in raw rows and aggregates;
12. semantic tamper tests fail closed after re-hash;
13. registry cannot clear confirmatory blockers or claim EXP-290/generalization/lemma-economy evidence;
14. tiny CLI execution passes in GitHub Actions;
15. all existing EXP-277/279/282/286 regressions remain green;
16. frozen Stage-A protocol files are absent from the PR diff;
17. package/runtime versions agree at `0.14.0` only after the full lane is green;
18. exact-head `core (3.11)`, `core (3.13)` and `model-smoke` are green;
19. review has no unresolved Critical/Important issue;
20. squash merge uses exact expected-head protection and post-merge `main` CI is verified before engineering closure is claimed.

## 20. What EXP-289 cannot establish

Even a strong DEVELOPMENT result cannot establish that:

- learned clauses generalize across surface-randomized instances;
- a neural memory can retrieve structurally equivalent nogoods safely;
- long-lived lemma libraries remain net-positive;
- clause garbage collection is safe;
- truth-maintenance under premise retraction is solved;
- full V0.16 beats simpler 100M rivals.

Those remain separate downstream hypotheses/experiments. EXP-289 is deliberately narrow: **can exact, sound, episode-local failure reuse prevent enough repeated dead-end work to justify proceeding?**
