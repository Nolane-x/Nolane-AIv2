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

`local_nogood` may store only dead-end partial assignments that its own search has actually reached. `no_nogood` executes the same memory-adapter/null path and accounting hooks but never persists or applies a nogood.

This isolates the frozen question: **does exact local failure reuse reduce repeated dead-end work enough to justify its memory/lookup cost without pruning valid solution states?**

### Rejected alternative A: direct reuse of EXP-279 scientific-arm authority

EXP-279 established a matched propagation/branch/hybrid routing court, but binding EXP-289 directly to that experiment's scientific artifact would couple two frozen hypotheses. EXP-289 may reuse low-level utilities and patterns, not EXP-279 evidence authority.

### Rejected alternative B: neural associative nogood memory

A learned memory would confound whether nogoods are useful, whether retrieval can learn structural equivalence, and whether approximate retrieval is safe. Those questions belong downstream, especially EXP-290.

### Rejected alternative C: keep only the existing algorithmic proxy

The Stage-A harness already uses `EpisodeNogoodStore` with repeated search calls. That is useful `EV-E2_ALGORITHMIC_PROXY` smoke evidence, but it does not close matched neural parameters, deterministic restart opportunities, maintenance-cost accounting, exact over-prune audit, artifact provenance or registry integration.

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
- pays the shared neural adapter cost;
- does not pay fictitious store insertion/subset-match costs that it did not execute.

### 4.3 `local_nogood`

The treatment arm owns an exact store scoped to one episode.

It may store a nogood only when its **observable search protocol** emits a local logical-dead-end event and:

1. the current partial assignment was reached through the arm's actual search path;
2. the dead-end event is produced by the same constraint/propagation/search semantics available to the arm, not by hidden evaluator truth;
3. the partial assignment is non-empty;
4. the assignment is canonicalized deterministically;
5. the store scope matches the current `episode_digest` and `problem_digest`.

The evaluator must **not gate insertion** using ground-truth solution enumeration, oracle conflict cores or future-path knowledge. Otherwise the treatment arm would receive an unearned oracle safety filter and the over-prune endpoint would become vacuous.

A stored nogood is an exact conjunction of variable assignments. It matches a later partial state only when the stored assignment is a subset of the current assignment. No approximate similarity retrieval is allowed in EXP-289.

## 5. Episode scope and leakage boundary

Every store binds to an exact scope receipt:

```json
{
  "scope": "episode_local",
  "episode_digest": "...",
  "problem_digest": "...",
  "cross_episode_reuse": false,
  "cross_problem_reuse": false,
  "oracle_conflict_core_used": false,
  "ground_truth_safety_gate_used": false,
  "learned_clause_generalization_claimed": false
}
```

The store is created at episode start and destroyed at episode end.

The semantic validator rejects semantic/provenance drift such as:

- reuse from another episode or problem digest;
- empty/root keys where the store policy forbids them;
- insertion before the arm-observable dead-end event;
- oracle conflict-core, solution or future-path information entering the key;
- evaluator ground truth being fed back to gate insertion;
- any claim that exact episode-local reuse demonstrates structural transfer.

## 6. Soundness audit without oracle assistance

EXP-289 must separate **arm behavior** from **evaluation truth**.

The local arm decides to insert from its observable dead-end event only. After the insertion has happened, an evaluator with generator truth independently asks whether the stored partial assignment has any valid complete extension.

For bounded DEVELOPMENT worlds, the evaluator must be able to enumerate or deterministically verify valid completions.

Every insertion receipt records:

- canonical nogood key;
- insertion step/restart index;
- arm-observable dead-end evidence digest;
- scope digests;
- post-hoc evaluator result `has_valid_completion`;
- charged canonicalization/insertion cost.

The `has_valid_completion` result is evaluator-only evidence. It must never be returned to the arm/controller during the episode.

An unsound stored key is a **scientific/safety outcome**, not an infrastructure-invalid artifact. The validator must preserve and correctly recompute such a failure; it must not silently reject/delete the row merely because the experiment performed badly. The validator rejects only false or inconsistent reporting of that failure.

## 7. Development world and restart generator

Create deterministic globally solvable worlds designed to expose repeat opportunities across bounded restarts without giving either arm privileged knowledge.

Each episode includes:

- a finite-domain CSP/search world with at least one verified solution;
- decoy regions capable of producing logically equivalent dead-end partial assignments under different restart orders;
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
- `controller_noise` only if a predeclared identical controller-noise path is actually required.

No confirmatory or challenge randomness may be derived.

The same replicate must regenerate byte-identical world tensors, restart schedule and evaluator opportunity metadata for both arms.

## 8. Repeat-dead-end opportunity contract

The primary endpoint must not use a denominator that either arm can shrink by altering its own path.

For every episode, the generator freezes an evaluator-only **opportunity manifest** before arm execution. The manifest is produced by a deterministic reference traversal that does not use either learned arm or any nogood store. Each item binds:

- restart index;
- canonical dead-end equivalence key;
- earlier occurrence index of the same key;
- world/restart lineage digest.

A manifest item is a repeat opportunity if the canonical reference traversal reaches a dead-end key that already appeared in an earlier restart.

The manifest is never exposed to either arm during execution.

The DEVELOPMENT primary endpoint is:

```text
repeat_dead_end_rate = repeated_dead_end_reentries / predeclared_repeat_opportunities
```

where:

- `predeclared_repeat_opportunities` comes only from the frozen opportunity manifest;
- `repeated_dead_end_reentries` counts manifest-equivalent repeated states actually entered by the arm instead of being avoided before full dead-end cost;
- because both arms share neural initialization/policy and exogenous restart lineage, post-hit path divergence is retained as a causal consequence of the memory intervention;
- preventing a repeat through a sound local-store match lowers the numerator but cannot alter the denominator.

If the manifest contains zero repeat opportunities, the episode remains in raw evidence with `rder_defined=false`. It contributes to safety/solution/cost metrics but not to the RDER numerator or denominator. The aggregate must explicitly report how many such episodes exist.

Descriptive receipts also report:

- `prevented_repeat_count`;
- `nogood_hits`;
- `unique_dead_end_count`;
- raw dead-end signatures;
- per-restart dead-end lineage;
- opportunity-manifest digest and count.

The frozen confirmatory bootstrap/MESI rule is not executed in DEVELOPMENT.

## 9. Over-prune safety court

`valid_state_overprune_rate` must be independently auditable and must not be made trivially safe by an oracle insertion filter.

Every actual `local_nogood` prune event is checked post hoc against regenerated world truth. A prune is invalid if the pruned partial state has at least one valid complete extension.

The DEVELOPMENT protected metric is defined operationally as:

```text
valid_state_overprune_rate = invalid_nogood_prune_events / max(nogood_prune_events, 1)
```

If no nogood prune event occurs, the rate is `0.0` and `nogood_prune_events=0` is reported explicitly; the primary RDER then prevents a no-op memory from looking useful.

The evaluator also reports:

- `store_soundness_violation_count`: stored keys with at least one valid completion;
- raw offending store/prune/state digests;
- number of valid solutions/prefixes checked.

A nonzero over-prune or soundness-violation result is valid negative scientific evidence. The semantic validator must recompute and preserve it, not convert it into an invalid-run classification unless provenance itself is corrupted.

The frozen protected floor remains `<=0.005`; DEVELOPMENT records the metric but does not execute the confirmatory promotion/kill rule.

## 10. Search, restart and causal-divergence semantics

Pairing freezes:

- initial world;
- model initialization;
- restart schedule;
- exogenous RNG lineage;
- maximum episode budget;
- evaluator opportunity manifest.

Pairing does **not** require post-intervention trajectories to remain identical. Avoiding a dead end because of a stored nogood is the intended causal effect.

For each restart:

1. both arms receive the same restart initial state/order;
2. each executes its own matched neural search path;
3. arm-observable dead-end events are recorded;
4. baseline records but never persists them;
5. local arm may insert exact non-empty keys from its observable dead ends;
6. later local queries may prune exact subset matches in the current episode store;
7. evaluator truth audits insertion/prune safety only after the arm action and never feeds back into control;
8. search continues until verified solution or the shared episode ceiling is exhausted.

Scientific/search failures are retained. Selective reruns are forbidden except predeclared infrastructure failures.

## 11. Parameter and resource matching

The matched-pair audit reports at minimum:

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

The symbolic store is experimental state, not free neural parameters. Its canonicalization, storage and lookup work is charged explicitly when executed.

## 12. Accounted-cost ledger

EXP-289's primary endpoint is RDER, but frozen resource matching still requires a deterministic cost court.

The analytical ledger separates at least:

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

All path-dependent cost is accumulated from raw step receipts. No hardware-profiler FLOP claim is permitted.

If an episode exhausts the common budget:

- it remains a scientific outcome;
- solution failure is recorded;
- cost is censored at the common ceiling;
- raw restart/search/store receipts remain present.

## 13. Training and evaluation

Training and evaluation use disjoint replicate lineages.

Training:

- `augmentation` stream;
- identical paired initialization;
- same optimizer family/hyperparameters/update count;
- no evaluation/confirmatory data mixed into training;
- exact store semantics remain episode-local;
- no cross-episode learned memory is introduced.

Development evaluation:

- `evaluation` stream;
- replicate range disjoint from training;
- shared paired worlds/restart schedules/opportunity manifests;
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
- evaluator opportunity-manifest receipts;
- post-hoc soundness/over-prune receipts;
- descriptive aggregates;
- explicit remaining blockers;
- canonical artifact self-hash.

The validator independently regenerates or recomputes critical semantics and rejects at least:

- arm/protocol/endpoint/MESI/protected-floor drift;
- initial-state or parameter mismatch;
- training/evaluation lineage overlap;
- episode/problem scope drift;
- cross-episode store reuse;
- empty/root insertion contrary to the policy;
- insertion before an arm-observable dead-end event;
- evaluator/oracle truth fed back into arm control;
- approximate/fuzzy matching masquerading as exact local retrieval;
- opportunity-manifest or denominator drift;
- repeated-dead-end numerator drift;
- deleted/reordered dead-end or restart rows;
- forged `prevented_repeat_count`, `nogood_hits`, prune counts or store counts;
- false reporting of soundness/over-prune outcomes;
- candidate/verified solution drift;
- storage/retrieval cost omission;
- per-step or episode accounted-cost drift;
- ceiling/censoring drift;
- cross-episode generalization claims;
- confirmatory/challenge/promotion fields set true;
- remaining-blocker removal;
- artifact digest mismatch.

Critically, the validator must **accept faithfully recorded negative scientific outcomes** such as nonzero over-prune, unsound stored keys, low RDER benefit or failed solution rate. Validation means the artifact is truthful and reconstructable, not that EXP-289 succeeded scientifically.

The validator must still reject semantic tampering after an attacker recomputes the top-level self-hash.

## 15. Neural-arm registry integration

Extend `src/nolane_ai/experiments/neural_arm_registry.py` to include `EXP-289` while preserving exact frozen arm descriptions.

Add optional evidence inputs:

- `exp289_pair_audit`;
- `exp289_execution_artifact`.

A valid pair audit plus paired DEVELOPMENT artifact may advance an engineering status such as `PAIRED_LOCAL_NOGOOD_DEV_READY` regardless of whether descriptive scientific effect is positive or negative; the status means the court executed validly, not that the hypothesis passed.

`match_court` remains `BLOCKED` because confirmatory sample-size/analysis freeze, confirmatory-open execution and post-freeze challenge evidence remain outstanding.

Registry language must not imply:

- EXP-290 learned-clause transfer is validated;
- lifelong lemma reuse is validated;
- conflict-core localization is validated;
- confirmatory H-CONFLICT-01 evidence exists.

## 16. Relationship to existing `EpisodeNogoodStore`

`src/nolane_ai/reasoning/search.py` already contains `EpisodeNogoodStore` and algorithmic branch/hybrid hooks.

EXP-289 may reuse exact subset-match semantics or refactor common exact-store primitives if required, but experiment-specific development state must add:

- explicit episode/problem scope;
- insertion-event provenance;
- deterministic canonical serialization;
- maintenance-cost receipts;
- safe reset/destruction boundary;
- validator-reconstructable lineage.

No unrelated rewrite of general search is required.

## 17. CLI, CI and version boundary

Add a CPU-safe development CLI:

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

## 18. Expected implementation files

Production:

- `src/nolane_ai/experiments/matched_nogood_arms.py`
- `src/nolane_ai/experiments/exp289_nogood_worlds.py`
- `src/nolane_ai/experiments/exp289_paired_runner.py`
- `scripts/run_exp289_paired_dev.py`
- modifications to `src/nolane_ai/experiments/neural_arm_registry.py`;
- CI/package/README updates;
- only narrowly scoped common-search/store refactoring if required by provenance/accounting.

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
5. insertion uses only arm-observable dead-end evidence; evaluator ground truth never gates arm behavior;
6. baseline cannot persist/apply a real nogood;
7. opportunity denominator is arm-independent and validator-regenerated;
8. repeat re-entry/prevention receipts are reconstructable from raw restart/search traces;
9. over-prune and stored-key soundness are independently recomputed from regenerated ground truth;
10. negative safety/scientific outcomes remain faithfully representable as valid artifacts;
11. storage/canonicalization/retrieval costs are charged and cannot be edited away;
12. unresolved/censored scientific outcomes remain in raw rows and aggregates;
13. semantic tamper tests fail closed after re-hash;
14. registry cannot clear confirmatory blockers or claim EXP-290/generalization/lemma-economy evidence;
15. tiny CLI execution passes in GitHub Actions;
16. all existing EXP-277/279/282/286 regressions remain green;
17. frozen Stage-A protocol files are absent from the PR diff;
18. package/runtime versions agree at `0.14.0` only after the full lane is green;
19. exact-head `core (3.11)`, `core (3.13)` and `model-smoke` are green;
20. review has no unresolved Critical/Important issue;
21. squash merge uses exact expected-head protection and post-merge `main` CI is verified before engineering closure is claimed.

## 20. What EXP-289 cannot establish

Even a strong DEVELOPMENT result cannot establish that:

- learned clauses generalize across surface-randomized instances;
- a neural memory can retrieve structurally equivalent nogoods safely;
- long-lived lemma libraries remain net-positive;
- clause garbage collection is safe;
- truth-maintenance under premise retraction is solved;
- full V0.16 beats simpler 100M rivals.

Those remain separate downstream hypotheses/experiments. EXP-289 is deliberately narrow: **can exact, scoped, episode-local failure reuse prevent enough repeated dead-end work to justify proceeding?**
