# EXP-279 V8 Powered Support Stability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the final power-sized augmentation-only EXP-279 branch-rescue support replication, with global 96-fold family-wise sizing, exact provenance, and no automatic post-visibility reruns.

**Architecture:** V8 starts from clean `main` and creates an isolated support-audit implementation. Eight canonical-root shard jobs measure only forced-stop/forced-branch sufficient statistics on fresh augmentation roots. Pure reducers aggregate four shards per train budget and a cross-cell classifier applies the frozen `MODEL_ROOT_SUPPORT_COLLAPSE → PROBE_SUPPORT_INTERMITTENT → SUPPORT_RECURRENT` hierarchy. A contract workflow is separated from a release-marker-triggered scientific workflow so later engineering commits cannot silently rerun powered data.

**Tech Stack:** Python 3.11, PyTorch CPU, pytest, GitHub Actions, existing EXP-279 canonical training/helpers.

**Spec:** `docs/superpowers/specs/2026-09-12-exp279-powered-support-stability-v8-design.md`

## Global Constraints

- Base is exact `main@803fb474eca9cf57713f190e89ef17a113f335e5`.
- Frozen Stage-A protocol digest is `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Evidence remains `EV-E2 / UNVERIFIED`.
- Decision budgets are exactly `60` and `120`.
- Canonical roots per budget: `4`; probe roots per canonical: `4`; diagnostic folds per probe: `3`.
- Powered scientific probe geometry is exactly `1332` replicates/probe root × batch `8` = `10656` episodes/probe root; `3552` episodes/fold; `42624` episodes/canonical root.
- Power input is frozen V7 floor `0.0025814897747382503`; global fold family is `96`; family-wise zero-support bound is `0.01`.
- Canonical route threshold `0.5`; d_model `64`; hidden `48`; target params `500000`; T `4`; variables `6`; constraints `3`; noise `0.05`; LR `0.002`; weight decay `0`.
- Training and probes use augmentation RNG only. Evaluation, confirmatory, challenge, promotion, and `60000..60032` remain locked.
- V8 is the final sample-size escalation for this support question. Negative V8 may not authorize “same court with more samples”.
- No selector, CDD, preview router, calibration model, or threshold tuning.

---

### Task 1: Core Power/Support Contract — RED

**Files:**
- Create: `tests/test_exp279_powered_support_stability_v8.py`

**Interfaces:**
- Consumes later production module `nolane_ai.experiments.exp279_powered_support_stability_v8`.
- Freezes constants/functions later tasks must preserve.

- [ ] **Step 1: Write failing core tests**

Tests must import and freeze:

```python
from nolane_ai.experiments.exp279_powered_support_stability_v8 import (
    FROZEN_V7_FLOOR,
    FROZEN_GLOBAL_FOLDS,
    FROZEN_FAMILY_ALPHA,
    PROBE_REPLICATES,
    PROBE_FOLDS,
    ROOT_PREFIX,
    required_fold_episodes,
    diagnostic_fold,
    wilson_lower_bound,
    expected_root_map,
    summarize_probe_support,
    summarize_canonical_support,
    classify_train_cell,
    run_exp279_powered_support_stability_v8_shard,
    validate_v8_shard,
)
```

Freeze:

```python
assert FROZEN_V7_FLOOR == 0.0025814897747382503
assert FROZEN_GLOBAL_FOLDS == 96
assert FROZEN_FAMILY_ALPHA == 0.01
assert required_fold_episodes() == 3552
assert PROBE_REPLICATES == 1332
assert PROBE_FOLDS == 3
```

Also assert:
- every scientific root string begins `20260912-exp279-powered-support-v8-dev`;
- train60/train120 root sets are disjoint;
- each root map contains exactly 4 canonical + 16 probe roots;
- `diagnostic_fold` gives exactly 444 replicates/fold when `range(1332)` is counted;
- support summaries recompute counts/booleans/Wilson bounds from fold counts;
- train-cell priority is collapse first, intermittent second, recurrent only with complete binary support;
- tiny shard exports only sufficient statistics and locked false boundaries;
- scientific validator rejects wrong protocol, geometry, sample count, root identity, commit provenance, and any evaluation/confirmatory/promotion boundary violation.

- [ ] **Step 2: Run RED**

Run:

```bash
pytest -q tests/test_exp279_powered_support_stability_v8.py
```

Expected: collection fails with `ModuleNotFoundError: nolane_ai.experiments.exp279_powered_support_stability_v8`.

- [ ] **Step 3: Commit test-only RED**

```bash
git add tests/test_exp279_powered_support_stability_v8.py
git commit -m "test(exp279): freeze V8 powered support contract"
```

---

### Task 2: Core Powered Support Implementation — GREEN

**Files:**
- Create: `src/nolane_ai/experiments/exp279_powered_support_stability_v8.py`
- Test: `tests/test_exp279_powered_support_stability_v8.py`

**Interfaces:**
- Produces `SCHEMA_SHARD`, `SCHEMA_BUDGET`, frozen constants, root helpers, power calculation, support summaries, `run_exp279_powered_support_stability_v8_shard(...)`, `validate_v8_shard(...)`, and `aggregate_exp279_powered_support_stability_v8_budget(shards)`.

- [ ] **Step 1: Implement frozen constants and power calculation**

Use exact constants:

```python
FROZEN_V7_FLOOR = 0.0025814897747382503
FROZEN_GLOBAL_FOLDS = 96
FROZEN_FAMILY_ALPHA = 0.01
PROBE_REPLICATES = 1332
PROBE_FOLDS = 3
ROOT_PREFIX = "20260912-exp279-powered-support-v8-dev"
```

Implement:

```python
def required_fold_episodes(
    p_floor: float = FROZEN_V7_FLOOR,
    global_folds: int = FROZEN_GLOBAL_FOLDS,
    family_alpha: float = FROZEN_FAMILY_ALPHA,
    batch_size: int = 8,
) -> int:
    raw = math.ceil(math.log(family_alpha / global_folds) / math.log(1.0 - p_floor))
    return math.ceil(raw / batch_size) * batch_size
```

Require result `3552`.

- [ ] **Step 2: Implement fresh V8 roots and pure support math**

Provide `canonical_root`, `probe_root`, `expected_root_map`, `diagnostic_fold`, exact Wilson lower bound, outcome-count reducer, probe/canonical summary, and train-cell classification. Do not import or use V7 artifacts at runtime.

- [ ] **Step 3: Implement shard executor using existing canonical EXP-279 training helpers**

Train one canonical hybrid on its V8 canonical augmentation root, freeze parameters, then run four V8 augmentation probe roots. Compute same-weight forced stop/forced branch exact outcomes only. Receipt must include:

```python
"scientific_branch_head": scientific_branch_head,
"executed_commit": executed_commit,
"code_digest": code_digest,
"evaluation_rng_stream_used": False,
"evaluation_targets_used": False,
"confirmatory_data_consumed": False,
"fresh_evaluation_lineage_may_be_reserved": False,
"fresh_evaluation_lineage_consumed": False,
"challenge_materialized": False,
"promotion_claimed": False,
```

No raw examples/model outputs.

- [ ] **Step 4: Implement absolute validator and pure budget reducer**

Scientific validator must require exact frozen configuration and 40-hex provenance fields. Budget reducer must require exactly canonical indices `{0,1,2,3}`, identical branch-head/executed-commit/code digest/protocol/config, exact fresh root map, 16 pair records, 48 fold records, and total episodes `170496`.

Aggregate receipt must carry `scientific_branch_head`, `executed_commit`, source shard artifact digests, root-map digest, prevalence floor, and classification.

- [ ] **Step 5: Run GREEN**

```bash
pytest -q tests/test_exp279_powered_support_stability_v8.py
python scripts/verify_protocol.py
python -m compileall -q src
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/nolane_ai/experiments/exp279_powered_support_stability_v8.py tests/test_exp279_powered_support_stability_v8.py
git commit -m "feat(exp279): add V8 powered support core"
```

---

### Task 3: Shard/Budget CLI — RED then GREEN

**Files:**
- Create: `tests/test_exp279_powered_support_stability_v8_cli.py`
- Create: `scripts/run_exp279_powered_support_stability_v8_dev.py`
- Create: `scripts/aggregate_exp279_powered_support_stability_v8_dev.py`

**Interfaces:**
- Runner writes one shard JSON.
- Aggregator consumes exactly four shard JSONs and writes one budget JSON.

- [ ] **Step 1: Write CLI RED tests**

Freeze these behaviors:
- output-existing rejection happens before protocol/input work;
- runner requires `--train-replicates`, `--canonical-index`, `--scientific-branch-head`, `--executed-commit`, `--output`;
- production defaults are frozen scientific geometry and `1332` replicates;
- `--tiny-contract` may reduce model/probe geometry for tests only;
- `--help` exposes no evaluation/confirmatory/challenge/fresh-lineage knobs;
- canonical protocol digest mismatch fails closed;
- aggregator rejects duplicate/missing canonical indices, wrong roots, provenance mismatch, and non-frozen scientific shards;
- deterministic stdout contains artifact digest only plus minimal identity.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/test_exp279_powered_support_stability_v8_cli.py
```

Expected: failures because both scripts do not exist.

- [ ] **Step 3: Implement minimal runner/aggregator**

Runner verifies canonical frozen protocol and computes source-tree digest before executing the core court. Aggregator only loads JSON after refusing existing output and delegates all science to the pure reducer.

- [ ] **Step 4: Verify GREEN and commit**

```bash
pytest -q tests/test_exp279_powered_support_stability_v8.py tests/test_exp279_powered_support_stability_v8_cli.py
python scripts/verify_protocol.py
python -m compileall -q src scripts
```

Commit:

```bash
git add tests/test_exp279_powered_support_stability_v8_cli.py scripts/run_exp279_powered_support_stability_v8_dev.py scripts/aggregate_exp279_powered_support_stability_v8_dev.py
git commit -m "feat(exp279): add V8 powered support CLIs"
```

---

### Task 4: Cross-Cell/Provenance Court — RED then GREEN

**Files:**
- Create: `tests/test_exp279_powered_support_stability_v8_cross_cell.py`
- Create: `src/nolane_ai/experiments/exp279_powered_support_stability_v8_cross_cell.py`
- Create: `scripts/classify_exp279_powered_support_stability_v8_dev.py`

**Interfaces:**
- `classify_exp279_powered_support_stability_v8_cross_cell(cells: dict[int, dict[str, Any]]) -> dict[str, Any]`.

- [ ] **Step 1: Write classifier RED tests**

Synthetic budget receipts must exercise:
- either-cell collapse → cross-cell `MODEL_ROOT_SUPPORT_COLLAPSE`;
- no collapse + either intermittent → `PROBE_SUPPORT_INTERMITTENT`;
- both recurrent → `SUPPORT_RECURRENT`;
- negative results set `successor_design_authorized=false`, `mechanism_successor_authorized=false`, `authorization_scope="CANONICAL_BRANCH_SEMANTICS_RESEARCH_ONLY"`;
- recurrent sets `successor_design_authorized=true`, `mechanism_successor_authorized=false`, `authorization_scope="DESIGN_NEW_MECHANISM_COURT_ONLY"`;
- no disposition ever reserves/consumes `60000..60032` or confirmatory/challenge/promotion data;
- exact train60/train120 required;
- classifier recomputes cell classification from sufficient statistics rather than trusting receipt label;
- branch-head, executed-commit, code digest, protocol, geometry, root maps, and sample counts are absolute/frozen;
- mutate both cells to the same wrong identity and require rejection;
- no prospective “increase sample again” recommendation exists in negative outcomes.

- [ ] **Step 2: Verify RED**

Expected missing cross-cell module.

- [ ] **Step 3: Implement classifier, then verify module tests before creating CLI**

Preserve exact decision priority. Receipt must include both input artifact digests and exact provenance.

- [ ] **Step 4: Observe CLI-only RED, implement CLI, verify GREEN**

```bash
pytest -q tests/test_exp279_powered_support_stability_v8_cross_cell.py
```

Then commit module + CLI after full pass.

---

### Task 5: Workflow and Freeze-Safety Contract — RED then GREEN

**Files:**
- Create: `tests/test_exp279_powered_support_stability_v8_workflow.py`
- Create: `.github/workflows/exp279-powered-support-stability-v8-ci.yml`
- Create: `.github/workflows/exp279-powered-support-stability-v8-scientific.yml`
- Create: `.github/workflows/exp279-powered-support-stability-v8-freeze-guard.yml`

**Interfaces:**
- Contract workflow runs V8 tests only.
- Scientific workflow triggers only on addition/change of `protocols/exp279-v8-powered-support-release.lock`.
- Freeze guard prevents any non-authoritative scientific rerun after first visibility.

- [ ] **Step 1: Write static workflow RED tests**

Assert:
- contract workflow uses CPU-only torch and includes all V8 tests;
- scientific workflow `pull_request.paths` contains only `protocols/exp279-v8-powered-support-release.lock`;
- scientific workflow contains matrix `[60,120] × [0,1,2,3]` and hard-coded `1332` probe replicates;
- shard runner receives `${{ github.event.pull_request.head.sha }}` as branch head and `${{ github.sha }}` as executed commit;
- reducers install CPU-only torch;
- budget/cross-cell jobs require shard success and verify JSON sidecars;
- scientific artifacts retain 90 days;
- freeze guard has `actions: write`, identifies the authoritative first scientific run, and cancels later V8 scientific runs rather than consuming their outputs.

Expected RED because workflows are missing.

- [ ] **Step 2: Implement workflows**

Do not create release marker yet. Scientific workflow must exist but remain dormant until marker commit.

- [ ] **Step 3: Verify workflow GREEN**

```bash
pytest -q tests/test_exp279_powered_support_stability_v8_workflow.py
```

Commit workflow files and tests.

---

### Task 6: Dedicated Contract-Only Gate

**Files:**
- No scientific marker yet.

- [ ] **Step 1: Confirm dedicated contract workflow has only engineering tests executing**

Verify GitHub Actions shows no powered shard job/run from V8 scientific workflow.

- [ ] **Step 2: Require GREEN**

Required before release:
- V8 dedicated contract SUCCESS;
- EXP-279 focused confirmatory CI SUCCESS;
- protocol verifier SUCCESS;
- compile SUCCESS.

If any fails, fix only pre-data engineering and rerun. Do not create release marker.

---

### Task 7: Scientific Release Marker and Frozen Matrix

**Files:**
- Create: `protocols/exp279-v8-powered-support-release.lock`

- [ ] **Step 1: Create the one-time release marker only after Task 6 GREEN**

Content:

```text
EXP279_V8_POWERED_SUPPORT_RELEASE_V1
p_floor=0.0025814897747382503
global_folds=96
family_alpha=0.01
fold_episodes=3552
probe_replicates=1332
```

This commit is the V8 scientific branch head passed into receipts.

- [ ] **Step 2: Freeze scientific surfaces immediately after first shard result visibility**

No edits to core, roots, sample size, predicates, Wilson math, disposition logic, or release marker after first result.

- [ ] **Step 3: Let 8 shards → 2 budget reducers → 1 cross-cell reducer complete**

Do not manually choose roots or reinterpret cells. Automated cross-cell receipt is authority.

---

### Task 8: Artifact Integrity and Scientific Interpretation

**Files:**
- Update PR body only; do not alter scientific code.

- [ ] **Step 1: Capture all artifact IDs and GitHub ZIP SHA256 digests**

Require 8 shard artifacts, 2 budget artifacts, 1 cross-cell artifact.

- [ ] **Step 2: Independently verify every receipt `.sha256` sidecar**

Also verify canonical receipt artifact digests and exact provenance fields.

- [ ] **Step 3: Interpret only automated disposition**

- `MODEL_ROOT_SUPPORT_COLLAPSE`: close without merge; no sample escalation; next seam canonical training/branch semantics.
- `PROBE_SUPPORT_INTERMITTENT`: close without merge; no sample escalation; next seam canonical branch heterogeneity/stability.
- `SUPPORT_RECURRENT`: close diagnostic court without merging deployable behavior; only authorize separately designed mechanism court.

No path may touch `60000..60032`.

---

### Task 9: Final Exact-Head Verification and PR Disposition

**Files:**
- Update PR title/body.

- [ ] **Step 1: Run fresh exact-head verification**

Require:
- generic CI core Python 3.11 SUCCESS;
- generic CI core Python 3.13 SUCCESS;
- model-smoke SUCCESS;
- EXP-279 focused contract SUCCESS;
- frozen protocol verifier SUCCESS;
- dedicated V8 contract SUCCESS;
- freeze guard SUCCESS;
- `main` still exact `803fb474eca9cf57713f190e89ef17a113f335e5`.

- [ ] **Step 2: Write final scientific record**

Include TDD lineage, pre-data gate run, scientific release head, exact executed merge SHA, source-tree digest, all artifact IDs/ZIP hashes, receipt file hashes/digests, train60/train120 classifications, cross-cell disposition, and locked boundaries.

- [ ] **Step 3: Close PR without merge unless a later separately scoped product change is explicitly justified**

V8 is a DEVELOPMENT diagnostic. Do not merge diagnostic experiment machinery into `main` merely because the result is recurrent.
