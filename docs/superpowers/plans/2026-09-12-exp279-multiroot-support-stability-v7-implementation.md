# EXP-279 V7 Multi-Root Support Stability Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILLS: use test-driven-development for every production surface, systematic-debugging for unexpected failures, and verification-before-completion before any engineering/scientific completion claim. Keep all work DEVELOPMENT-only.

**Goal:** Implement the preregistered V7 audit that distinguishes canonical-model support collapse from probe-sampling intermittency across independent augmentation roots, without training a selector or touching fresh evaluation/confirmatory lineages.

**Architecture:** Split the scientific execution into eight independent canonical-root shards (`train budget ∈ {60,120}` × `canonical index ∈ {0,1,2,3}`). Each shard trains one canonical hybrid exactly once and evaluates four independent augmentation-only probe roots, emitting only sufficient statistics for 12 diagnostic folds. A pure budget reducer validates and aggregates exactly four canonical shards into one train-cell receipt. A pure cross-cell reducer validates the train60/train120 receipts, applies the preregistered disposition priority, and computes planning-only sample size when permitted. GitHub Actions first ships a contract-only workflow; only after that exact workflow contract is observed GREEN may the eight scientific shard jobs, two budget reducers, and one cross-cell reducer be added.

**Tech Stack:** Python 3.11, PyTorch CPU, pytest, GitHub Actions, existing EXP-279 generator/training/forced-path helpers, canonical JSON/SHA256 helpers.

**Spec:** `docs/superpowers/specs/2026-09-12-exp279-multiroot-support-stability-v7-design.md`

**Global Constraints:** Protocol `NLM-REASONING-STAGE-A-CONFIRMATORY-V1 / FROZEN_V1` digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`; `EV-E2 / UNVERIFIED`; no evaluation RNG/targets, confirmatory data, external examples, challenge materialization or promotion; `60000..60032` remains untouched; no selector/CDD/preview/threshold tuning; scientific roots/budgets/predicates freeze once first V7 result is visible.

## File Map

Create:
- `src/nolane_ai/experiments/exp279_multiroot_support_stability_v7.py` — frozen root naming, Wilson math, one-canonical-root execution, shard validation, budget aggregation/classification.
- `src/nolane_ai/experiments/exp279_multiroot_support_stability_v7_cross_cell.py` — absolute frozen-identity validation and train60/train120 disposition/sample-size planning.
- `scripts/run_exp279_multiroot_support_stability_v7_dev.py` — one canonical-root shard CLI.
- `scripts/aggregate_exp279_multiroot_support_stability_v7_dev.py` — four-shard budget reducer CLI.
- `scripts/classify_exp279_multiroot_support_stability_v7_dev.py` — train60/train120 cross-cell reducer CLI.
- `tests/test_exp279_multiroot_support_stability_v7.py` — pure/core and tiny execution contract.
- `tests/test_exp279_multiroot_support_stability_v7_cli.py` — CLI protocol/output/argument boundary.
- `tests/test_exp279_multiroot_support_stability_v7_cross_cell.py` — reducer priority, sample-size and absolute-identity tests.
- `.github/workflows/exp279-multiroot-support-stability-v7-ci.yml` — first contract-only, later scientific release DAG.

Reuse without modification:
- `src/nolane_ai/experiments/exp279_branch_rescue_oracle.py::_branch_outcome_counts`
- `src/nolane_ai/experiments/exp279_paired_runner.py::{_build_seeded_triplet,_functional_state_digest,_hybrid_stop_decision_logits,_hybrid_forced_branch_decision_logits,_train_step}`
- `src/nolane_ai/experiments/exp279_routing_worlds.py::{STRATA,Exp279RoutingGenerator}`
- `src/nolane_ai/experiments/matched_routing_arms.py::audit_matched_exp279_arm_triplet`
- `nolane_ai.protocol.evidence::{canonical_sha256,canonical_json_bytes}`
- existing protocol identity/verification helpers.

## Task 1 — Core RED: freeze support arithmetic, roots and shard schema

- [ ] Create `tests/test_exp279_multiroot_support_stability_v7.py` before any V7 production module exists.
- [ ] Import the future module at top level so the first RED is a clean `ModuleNotFoundError`.
- [ ] Freeze root naming exactly:

```python
assert canonical_root(60, 2) == (
    "20260912-exp279-multiroot-support-v7-dev::train::60::canonical::2"
)
assert probe_root(120, 3, 1) == (
    "20260912-exp279-multiroot-support-v7-dev::train::120::probe::3::1"
)
```

- [ ] Test invalid train budgets/indexes fail closed; only budgets `{60,120}`, canonical indices `0..3`, probe indices `0..3` are valid.
- [ ] Freeze diagnostic fold assignment:

```python
assert diagnostic_fold(0) == 0
assert diagnostic_fold(len(STRATA)) == 1
assert diagnostic_fold(2 * len(STRATA)) == 2
```

with formula `(replicate // len(STRATA)) % 3`.
- [ ] Freeze one-sided 95% Wilson implementation against hand-computed/reference values, including `x=0 -> 0.0`, `0 < x < n`, and invalid `x/n` rejection.
- [ ] Freeze exact outcome partition using existing `_branch_outcome_counts`: rescue/harm/both-success/both-fail must close to episode count.
- [ ] Freeze pure support summary from synthetic fold counts: probe support, fold support, pooled canonical count, pooled Wilson lower bound.
- [ ] Freeze train-cell priority with synthetic four-shard receipts:
  - any canonical total rescues `0` -> `MODEL_ROOT_SUPPORT_COLLAPSE`;
  - otherwise any probe/fold missing rescue or non-rescue -> `PROBE_SUPPORT_INTERMITTENT`;
  - otherwise -> `SUPPORT_RECURRENT`.
- [ ] Freeze root-map identity: exact 4 canonical roots and 16 probe roots for a budget, deterministic canonical SHA256, no duplicates.
- [ ] Add tiny shard execution (`train_replicates=3`, `probe_replicates=6`, tiny geometry) proving:
  - canonical training uses augmentation only;
  - every probe batch uses augmentation only under independent probe root;
  - no raw logits/examples are present in receipt;
  - final canonical digest is present;
  - exactly four probe receipts and twelve fold summaries are emitted;
  - evidence/lineage flags remain false.
- [ ] Commit RED and wait for GitHub `exp279-confirmatory-ci` to fail for the expected missing V7 module. Do not create production code before this evidence.

## Task 2 — Core GREEN: one-root executor + budget reducer

- [ ] Create `src/nolane_ai/experiments/exp279_multiroot_support_stability_v7.py` with constants:

```python
SCHEMA_SHARD = "NLM-EXP-279-MULTIROOT-SUPPORT-SHARD-V7"
SCHEMA_BUDGET = "NLM-EXP-279-MULTIROOT-SUPPORT-BUDGET-V7"
ROOT_PREFIX = "20260912-exp279-multiroot-support-v7-dev"
TRAIN_BUDGETS = (60, 120)
CANONICAL_INDICES = (0, 1, 2, 3)
PROBE_INDICES = (0, 1, 2, 3)
PROBE_REPLICATES = 198
PROBE_FOLDS = 3
WILSON_Z = 1.6448536269514722
```

- [ ] Implement root helpers and `diagnostic_fold(replicate)` with exact validation.
- [ ] Implement `wilson_lower_bound(rescues, episodes)` exactly from the spec, with finite/range checks.
- [ ] Implement a pure count reducer whose only categories are `branch_rescues`, `branch_harms`, `both_success`, `both_failure`; enforce closure before computing prevalence/support.
- [ ] Implement `_train_canonical_hybrid(...)` by replaying the existing canonical hybrid path:
  - `_build_seeded_triplet` under canonical root;
  - matched-resource audit must close;
  - optimizer built with existing functional optimizer;
  - `train_replicates` batches from `Exp279RoutingGenerator(canonical_root)` with `rng_stream="augmentation"` and canonical STRATA cycle;
  - `_train_step(... arm_id="hybrid")` only;
  - freeze/eval model after training.
- [ ] Implement `run_exp279_multiroot_support_stability_v7_shard(...)`:
  - validate budget/index/frozen settings;
  - train exactly one canonical model;
  - for each `P=0..3`, instantiate `Exp279RoutingGenerator(probe_root(N,C,P))`;
  - for replicate `0..probe_replicates-1`, use `rng_stream="augmentation"`, `STRATA[r % len(STRATA)]`;
  - under `torch.no_grad`, compute forced-stop and forced-branch logits with existing helpers;
  - call `_branch_outcome_counts`, assign the entire batch to `diagnostic_fold(r)`, aggregate only sufficient statistics;
  - never retain/export tensors, targets, logits, predictions or per-example masks.
- [ ] Receipt must explicitly carry:
  - `EV-E2`, `UNVERIFIED`, scientific ineligible;
  - protocol/code digest;
  - budget/index/root IDs;
  - exact frozen geometry/config;
  - training/probe RNG streams both `augmentation`;
  - `evaluation_rng_stream_used=false`, `evaluation_targets_used=false`;
  - selector/CDD/preview/threshold-tuned flags false;
  - fresh lineage/confirmatory/challenge/promotion flags false;
  - four probe sufficient-stat records + twelve fold records;
  - pooled canonical support and Wilson bound;
  - root-map fragment digest;
  - deterministic artifact digest excluding its own digest field.
- [ ] Implement `validate_v7_shard(receipt)` with absolute checks, not just self-consistency. It must reject wrong protocol ID/digest, wrong root prefix, wrong budget/index, wrong geometry, wrong probe count, wrong streams, duplicate roots, exported raw data flags, and lineage drift.
- [ ] Implement `aggregate_exp279_multiroot_support_stability_v7_budget(shards)`:
  - require exactly canonical indices `{0,1,2,3}` for one budget;
  - validate every shard independently;
  - require common code/protocol/frozen geometry;
  - require all canonical/probe roots unique and exact names;
  - compute 16 pair records and 48 fold records from shard sufficient statistics;
  - compute per-canonical pooled Wilson bounds and `canonical_prevalence_floor=min(...)`;
  - apply the exact priority classification;
  - emit root-map digest and artifact digest.
- [ ] Re-run focused tests on exact GREEN head; do not move to CLI until GitHub focused suite, protocol verifier and compile are GREEN.

## Task 3 — CLI RED → GREEN

- [ ] Create `tests/test_exp279_multiroot_support_stability_v7_cli.py` before scripts.
- [ ] RED requirements:
  - shard runner script absent initially;
  - budget aggregate script absent initially;
  - cross-cell script is not part of this RED yet.
- [ ] Shard CLI test requires only explicit DEVELOPMENT knobs; it must expose **no** `--eval-*`, confirmatory, challenge or fresh-lineage arguments.
- [ ] `--train-replicates` must accept only `60/120` for real mode; provide `--tiny-contract` only for test/contract execution, and ensure workflow scientific jobs never use it.
- [ ] `--canonical-index` must be 0..3; probe count fixed to four roots; scientific `probe_replicates` default/frozen at 198.
- [ ] Refuse overwrite before protocol/model work: pre-create output, pass a deliberately bad protocol path, assert error is `output already exists`.
- [ ] Protocol digest mismatch must fail closed against canonical Stage-A digest.
- [ ] Budget reducer CLI accepts exactly four `--shard` paths plus `--output`; rejects duplicate/missing canonical indices and overwrite.
- [ ] Observe GitHub RED: core tests remain GREEN, CLI tests fail only because scripts do not exist.
- [ ] Implement `scripts/run_exp279_multiroot_support_stability_v7_dev.py` using existing protocol verification pattern and `source_tree_digest(ROOT)`.
- [ ] Implement `scripts/aggregate_exp279_multiroot_support_stability_v7_dev.py`; it must load JSON only, invoke pure validated budget reducer, and write canonical sorted/indented JSON plus stdout compact summary.
- [ ] Re-run exact-head focused CI and compile to GREEN before cross-cell work.

## Task 4 — Cross-cell RED → GREEN + absolute frozen identity

- [ ] Create `tests/test_exp279_multiroot_support_stability_v7_cross_cell.py` before module/script.
- [ ] Freeze exact input budgets `{60,120}` and disposition priority:

```python
# collapse dominates everything
assert classify({60: collapse, 120: recurrent})["decision"] == "MODEL_ROOT_SUPPORT_COLLAPSE"
# intermittent dominates recurrent
assert classify({60: recurrent, 120: intermittent})["decision"] == "PROBE_SUPPORT_INTERMITTENT"
assert classify({60: recurrent, 120: recurrent})["decision"] == "SUPPORT_RECURRENT"
```

- [ ] Freeze authorization scope:
  - collapse: no mechanism/successor design, sample-size recommendation `None`;
  - intermittent: no mechanism successor; planning-only support-court sample size allowed if both canonical prevalence floors are positive;
  - recurrent: only design of a new mechanism court authorized; no mechanism implementation/evaluation lineage.
- [ ] Freeze sample-size arithmetic:

```python
n = math.ceil(math.log(0.01) / math.log(1.0 - p_floor))
```

and cross-cell future episodes = `max(n60, n120)`; any required floor `<=0` -> `None`.
- [ ] Absolute frozen-identity tests must mutate **both** budget receipts consistently to the same wrong value and still fail:
  - protocol digest;
  - route threshold;
  - geometry;
  - root prefix/naming;
  - canonical/probe root counts;
  - probe replicates/folds;
  - RNG stream;
  - evidence/decision flags;
  - lineage/confirmatory/challenge/promotion flags.
- [ ] Explicitly test train60/train120 root sets are disjoint.
- [ ] Commit and observe RED solely from missing cross-cell module.
- [ ] Implement `src/nolane_ai/experiments/exp279_multiroot_support_stability_v7_cross_cell.py` with shared-provenance validation first and absolute frozen-config validation second.
- [ ] Implement `scripts/classify_exp279_multiroot_support_stability_v7_dev.py`, refuse overwrite, validate both receipts, emit deterministic JSON/digest.
- [ ] Re-run exact-head focused CI GREEN. If any failure is unexpected, use systematic-debugging and preserve the test contract.

## Task 5 — Dedicated workflow contract-only pre-data gate

- [ ] Add `.github/workflows/exp279-multiroot-support-stability-v7-ci.yml` in **contract-only form**. It must have no scientific shard/reducer jobs yet.
- [ ] Trigger paths include the V7 modules/scripts/tests/spec/plan/workflow.
- [ ] `contract` installs CPU-only torch + dev deps, runs:

```bash
python scripts/verify_protocol.py
pytest -q \
  tests/test_exp279_multiroot_support_stability_v7.py \
  tests/test_exp279_multiroot_support_stability_v7_cli.py \
  tests/test_exp279_multiroot_support_stability_v7_cross_cell.py
python -m compileall -q src scripts
```

- [ ] Inspect workflow source on exact commit to prove no matrix/scientific job exists.
- [ ] Observe dedicated contract job GREEN on GitHub Actions.
- [ ] Do not add the scientific matrix until this exact pre-data contract run is GREEN.

## Task 6 — Scientific release workflow, frozen DAG

Only after Task 5 GREEN:

- [ ] Update the same workflow to add an 8-way shard matrix:

```yaml
strategy:
  fail-fast: false
  matrix:
    train_replicates: [60, 120]
    canonical_index: [0, 1, 2, 3]
needs: contract
```

- [ ] Each shard hard-codes scientific values: d_model64, hidden48, target500000, threshold0.5, probe replicates198, batch8, T4, V6, constraints3, noise.05, LR.002, WD0. No workflow inputs may override them.
- [ ] Each shard writes `receipt.json` and `receipt.json.sha256`, asserts all anti-leak flags, root naming, exactly 4 probes/12 folds/6336 episodes, and uploads a 90-day artifact named with budget/index and `${{ github.sha }}`.
- [ ] Add two budget reducer jobs (`budget-60`, `budget-120`) that `needs` all shard jobs, download exactly the four expected shard artifacts for their budget, verify embedded JSON SHA files before reduction, run budget reducer CLI, assert 16 pair/48 fold records and one of the three preregistered cell classifications, then upload receipt + SHA.
- [ ] Add `cross-cell-decision` requiring both budget reducers, download their receipts, verify SHA files, run cross-cell CLI, assert lineage/confirmatory/challenge/promotion false and `fresh_evaluation_lineage_may_be_reserved=false`, upload decision + SHA.
- [ ] Freeze this commit as the scientific release head. From the first visible V7 scientific result onward, do not modify V7 scientific code, roots, budgets, Wilson formula, support predicates or disposition logic on this lineage.

## Task 7 — Read evidence only; audit and disposition

- [ ] Let the automated reducer, not manual interpretation, decide the cross-cell disposition.
- [ ] Download all 8 shard artifacts, 2 budget artifacts and cross-cell artifact.
- [ ] Recompute each GitHub ZIP SHA256 independently; compare with GitHub artifact digest.
- [ ] For every archive, recompute JSON SHA256 and compare with embedded `.sha256`.
- [ ] Audit root uniqueness across all 8 canonical and 32 probe roots from receipts; verify no root intersects across budgets.
- [ ] Audit exact episode totals: each pair 1584, each canonical 6336, each budget 25344.
- [ ] Audit outcome partitions close for every pair/fold and that aggregate counts equal child sufficient statistics.
- [ ] Fetch fresh exact-release-head runs for:
  - dedicated V7 workflow;
  - `exp279-confirmatory-ci`;
  - generic `ci`.
  All must be SUCCESS before an engineering-clean completion claim.
- [ ] Update PR #57 body with frozen release SHA, automated disposition, per-budget support/Wilson summaries, root-map/artifact digests, artifact IDs/ZIP SHA256/embedded JSON SHA256, exact CI run IDs, and explicit statement that `60000..60032` remains untouched.
- [ ] Final disposition:
  - `MODEL_ROOT_SUPPORT_COLLAPSE`: close without merge; next seam must revisit canonical training stability/branch semantics/complementarity definition.
  - `PROBE_SUPPORT_INTERMITTENT`: close without merge; preserve planning-only prospective sample-size recommendation for a separately preregistered new-root support court.
  - `SUPPORT_RECURRENT`: V7 still does not authorize a router; preserve only authorization to design a separately preregistered mechanism court with new roots and V7-derived prospective sample size. Do not open `60000..60032`.

## Plan Self-Review Checklist

- [x] Every scientific quantity in the written spec has an implementation/test location.
- [x] No TODO/TBD/placeholders remain.
- [x] Train60 and train120 use disjoint root namespaces.
- [x] Zero-rescue canonical roots force Wilson prevalence floor to zero and dominate disposition.
- [x] Budget reducer receives sufficient statistics only; no raw examples cross jobs.
- [x] Workflow parallelism changes execution topology only, not the preregistered statistical unit or decision rule.
- [x] Contract-only workflow is a separate observed gate before data release.
- [x] No evaluation/confirmatory/fresh-lineage interface exists in the V7 CLI/workflow.
- [x] Scientific release becomes immutable after first result visibility.
