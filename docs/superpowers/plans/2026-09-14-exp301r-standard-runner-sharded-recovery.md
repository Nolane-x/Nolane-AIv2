# EXP-301R Standard-Runner Sharded Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the unchanged EXP-301 scientific court on ordinary `ubuntu-latest` runners by splitting independent trials and deterministic challenge prediction into auditable jobs that each fit below GitHub's six-hour hosted-job limit.

**Architecture:** Add a recovery-only orchestration layer beside, never inside, the frozen EXP-301 scientific implementation. Twenty-four independent frozen trial jobs feed four deterministic root-selection jobs; four challenge manifests feed sixteen prediction shards per root; four root sealers reconstruct the complete commitment grid and call the existing evidence builder; the unchanged cross-root reducer remains sole decision authority. Recovery receipts bind every derived artifact to the original frozen implementation, the failed run, the new workflow and one stable run-id beacon.

**Tech Stack:** Python 3.11, PyTorch, pytest, GitHub Actions, canonical JSON SHA-256 receipts.

**Spec:** `docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md`

## Global Constraints

- Original marker commit: `bac51c29c46e4c1fb3db5445a299da4674fdb6d8`.
- Original frozen implementation digest: `89ea87607b75061c0c3d426fdf07ba82cf1c0fb0ad62135c1106ccbaa5e8890c`.
- Prior failed run: `34823251660`; it has no uploaded artifacts and no scientific disposition.
- Do not modify existing EXP-301 scientific implementation/protocol/reducer/test files.
- CPU only; ordinary `ubuntu-latest`; no Enterprise/larger runner requirement.
- Preserve 4 roots, 3 arms, 2 LR trials/arm, 512 training steps/trial, 512 challenge worlds/root, efforts `(1,2,4,8,12,16)`, 96 challenge forwards/prediction, protected floors, bootstrap and reducer thresholds.
- Recovery challenge beacon is stable for a workflow `run_id` and does not contain `run_attempt`.
- Sixteen contiguous challenge world shards/root, exactly 32 worlds/shard.
- Shard jobs emit prediction commitments only and never score challenge answers.
- The unchanged EXP-301 reducer remains sole scientific decision authority.

---

### Task 1: Recovery core contracts and isolation verifier

**Files:**
- Create: `src/nolane_ai/experiments/exp301r_recovery.py`
- Create: `scripts/verify_exp301r_recovery.py`
- Create: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Produces: `recovery_beacon(run_id: str) -> str`, `challenge_shard_bounds(shard_index: int) -> tuple[int, int]`, `canonical_digest(payload: object) -> str`, `write_once_json(path, payload)`, recovery receipt dataclasses/loaders, and `verify_recovery_isolation(repo_root: Path) -> None`.
- Consumes only frozen EXP-301 public types/constants; does not mutate them.

- [ ] **Step 1: Write failing contract tests**

Tests must assert:

```python
assert recovery_beacon("34899900000") == "github-run-34899900000-exp301r-v1"
assert challenge_shard_bounds(0) == (0, 32)
assert challenge_shard_bounds(15) == (480, 512)
with pytest.raises(ValueError):
    challenge_shard_bounds(16)
assert tuple(challenge_shard_bounds(i) for i in range(16)) == tuple((i * 32, (i + 1) * 32) for i in range(16))
```

Also create a temporary Git repository rooted at the marker shape and prove the isolation verifier rejects a modification to an existing `src/nolane_ai/experiments/exp301_*.py` file while allowing newly added `exp301r_*`, `scripts/exp301r_*`, `tests/test_exp301r_*`, recovery docs and recovery workflow paths.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q`

Expected: import/attribute failure because `exp301r_recovery` and verifier do not exist.

- [ ] **Step 3: Implement minimal recovery core and isolation verifier**

Use canonical `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)` SHA-256. `challenge_shard_bounds` must hard-code exactly 16 shards over exactly 512 worlds. `write_once_json` must open with mode `x`.

`verify_recovery_isolation` must reject any path changed from marker `bac51c29...` outside these prefixes/exact files:

```text
.github/workflows/exp301r-standard-runner-sharded-recovery.yml
docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md
docs/superpowers/plans/2026-09-14-exp301r-standard-runner-sharded-recovery.md
src/nolane_ai/experiments/exp301r_
scripts/exp301r_
scripts/verify_exp301r_recovery.py
tests/test_exp301r_
```

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q`

Expected: all recovery-core tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nolane_ai/experiments/exp301r_recovery.py scripts/verify_exp301r_recovery.py tests/test_exp301r_recovery.py
git commit -m "feat(exp301r): add recovery identity and isolation contracts"
```

### Task 2: One frozen trial per standard-runner job

**Files:**
- Create: `scripts/exp301r_train_trial.py`
- Modify: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Consumes: `EXP301R_ROOT`, `EXP301R_ARM`, `EXP301R_TRIAL_INDEX`, `EXP301R_WORKFLOW_DIGEST`, `EXP301R_RUN_ID`, output directory, frozen identity path.
- Produces: canonical scientific checkpoint at existing `checkpoint_path_for_plan(...)` plus `EXP301R-TRIAL-RECEIPT-V1` JSON.

- [ ] **Step 1: Write failing tests**

Inject a fake trial runner and assert coordinate `(root=2, arm=C_NRS_CORE, trial_index=1)` resolves to exactly the existing `frozen_trial_plan(root=2)` entry. Assert arbitrary LR/seed input is impossible because the API accepts only frozen coordinate selectors. Assert receipt binds result checkpoint/trial digests, original frozen implementation digest, prior failed run id, run id and workflow digest.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k train_trial`

Expected: missing train-trial API.

- [ ] **Step 3: Implement minimal train-trial command**

Resolve the plan by exact frozen tuple, call existing `run_scientific_trial`, validate `training_steps == 512`, write one recovery receipt, and return zero. Never materialize challenge data.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q -k train_trial`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp301r_train_trial.py tests/test_exp301r_recovery.py src/nolane_ai/experiments/exp301r_recovery.py
git commit -m "feat(exp301r): shard frozen trials across hosted jobs"
```

### Task 3: Root selection from six audited trial artifacts

**Files:**
- Create: `scripts/exp301r_select_root.py`
- Modify: `src/nolane_ai/experiments/exp301r_recovery.py`
- Modify: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Consumes six canonical scientific checkpoint files for one root.
- Produces existing `selection-manifest.json`, only the three selected checkpoints in a selected-package directory, and `EXP301R-SELECTION-RECEIPT-V1`.

- [ ] **Step 1: Write failing tests**

Build six lightweight fake `ScientificTrialResult` values matching two frozen LRs for each arm. Assert missing, duplicate, wrong-root, wrong-seed, wrong-LR and digest-invalid trial evidence is rejected. Assert selected LR uses existing `build_arm_selection_receipt` tie rule and that exactly three selected checkpoint paths are exported.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k selection`

Expected: missing root-selection recovery API.

- [ ] **Step 3: Implement selection**

Load each real checkpoint through the existing scientific checkpoint validation path, build existing arm-selection receipts and `RootSelectionManifest`, use existing write-once manifest writer, copy only selected canonical checkpoints, and write a recovery selection receipt.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q -k selection`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp301r_select_root.py src/nolane_ai/experiments/exp301r_recovery.py tests/test_exp301r_recovery.py
git commit -m "feat(exp301r): seal root selection from audited trial shards"
```

### Task 4: Challenge manifest and prediction-only shards

**Files:**
- Create: `scripts/exp301r_materialize_challenge.py`
- Create: `scripts/exp301r_predict_shard.py`
- Modify: `src/nolane_ai/experiments/exp301r_recovery.py`
- Modify: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Challenge manifest produces `EXP301R-CHALLENGE-MANIFEST-V1` with root, beacon, nonce, materialization digest, ordered content IDs and digest.
- Prediction shard produces `EXP301R-PREDICTION-SHARD-V1` with exactly 576 existing `PredictionCommitment` objects and no evaluation rows.

- [ ] **Step 1: Write failing challenge-manifest tests**

Assert one root materializes exactly 512 ordered world IDs and rematerializing with the same `run_id` gives the exact same manifest identity. Assert changing only a hypothetical run-attempt value cannot affect beacon because attempt is absent from the API.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k challenge_manifest`

Expected: missing manifest API.

- [ ] **Step 3: Implement challenge manifest**

Call existing `materialize_root_challenge` with `recovery_beacon(run_id)`. Store identities/digests only.

- [ ] **Step 4: Write failing shard tests**

Inject a fake predictor and three fake selected models. Assert shard 0 consumes exactly worlds `[0:32]`, calls predictor `32 * 3 * 6 = 576` times, preserves canonical order `world -> arm -> effort`, every commitment has generation token count 96, and output contains no `verified_success`, `evaluation_rows`, or verifier score.

Assert shard manifest mismatch, wrong selected root, wrong challenge digest, overlap coordinate and out-of-range shard are rejected.

- [ ] **Step 5: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k prediction_shard`

Expected: missing shard API.

- [ ] **Step 6: Implement shard generation**

Rematerialize and match full challenge, slice with `challenge_shard_bounds`, call existing `scientific_challenge_generate`/`commit_prediction`, validate every commitment, and write one canonical shard receipt.

- [ ] **Step 7: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q -k 'challenge_manifest or prediction_shard'`

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/exp301r_materialize_challenge.py scripts/exp301r_predict_shard.py src/nolane_ai/experiments/exp301r_recovery.py tests/test_exp301r_recovery.py
git commit -m "feat(exp301r): add deterministic challenge prediction shards"
```

### Task 5: Complete-grid root sealing and recovery receipt

**Files:**
- Create: `scripts/exp301r_seal_root.py`
- Modify: `src/nolane_ai/experiments/exp301r_recovery.py`
- Modify: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Consumes selection manifest, challenge manifest and exactly 16 unique prediction shards for one root.
- Produces unchanged `root-evidence.json` through existing EXP-301 builders plus `EXP301R-ROOT-RECOVERY-RECEIPT-V1`.

- [ ] **Step 1: Write failing merge tests**

Assert the merger rejects 15 shards, 17 shards, duplicate index, foreign run id/beacon, foreign root, mismatched materialization digest and overlapping world IDs. Assert shuffled input shard files reconstruct commitments in canonical challenge/arm/effort order.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k seal_root`

Expected: missing root-seal API.

- [ ] **Step 3: Implement complete-grid sealing**

After all shard audits pass, rebuild runtime identity using existing `build_runtime_identity_for_root`, score with existing `score_challenge_commitments`, build with existing `build_root_evidence_artifact`, write with existing `write_root_evidence_artifact`, then write the recovery root receipt binding the root-evidence artifact digest and all shard digests.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q -k seal_root`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp301r_seal_root.py src/nolane_ai/experiments/exp301r_recovery.py tests/test_exp301r_recovery.py
git commit -m "feat(exp301r): seal complete sharded root evidence"
```

### Task 6: Cross-root recovery finalizer

**Files:**
- Create: `scripts/exp301r_finalize.py`
- Modify: `src/nolane_ai/experiments/exp301r_recovery.py`
- Modify: `tests/test_exp301r_recovery.py`

**Interfaces:**
- Existing `scripts/reduce_exp301.py` remains unchanged and produces `cross-root-evidence.json`.
- Finalizer consumes that file plus four recovery root receipts and produces `EXP301R-CROSS-ROOT-RECOVERY-ENVELOPE-V1` without recomputing decision semantics.

- [ ] **Step 1: Write failing tests**

Assert finalizer requires exactly roots 0..3, all recovery receipts bind the same frozen digest/run id/workflow digest/beacon, cross-root artifact digest is canonical, and final decision is copied verbatim from the reducer artifact. Assert `original_monolithic_run_completed` is exactly `False`, `prior_failed_run_id` is exactly `34823251660`, and no scale/implementation authorization can be introduced by finalizer fields.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py -q -k finalize`

Expected: missing finalizer API.

- [ ] **Step 3: Implement finalizer**

Audit four existing root evidences, bind their recovery receipts to artifact digests, load/audit the reducer output, copy its exact decision and authorization fields, and emit canonical write-once envelope.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_recovery.py -q -k finalize`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/exp301r_finalize.py src/nolane_ai/experiments/exp301r_recovery.py tests/test_exp301r_recovery.py
git commit -m "feat(exp301r): bind unchanged reducer outcome to recovery envelope"
```

### Task 7: Recovery GitHub Actions workflow and static contract

**Files:**
- Create: `.github/workflows/exp301r-standard-runner-sharded-recovery.yml`
- Create: `tests/test_exp301r_workflow.py`

**Interfaces:**
- Manual `workflow_dispatch` only.
- Uses `ubuntu-latest` and Python 3.11.
- Matrix: 24 trial jobs, 4 selection jobs, 4 challenge-manifest jobs, 64 prediction-shard jobs, 4 root-seal jobs, one cross-root reducer/finalizer job.

- [ ] **Step 1: Write failing workflow contract test**

Parse workflow text and assert:

```python
assert "workflow_dispatch" in text
assert "ubuntu-latest" in text
assert "EXP301_DEVICE: cpu" in text
assert "EXP301R_SHARD_COUNT: 16" in text
assert "run_attempt" not in recovery_beacon_lines
assert "python scripts/reduce_exp301.py" in text
assert "python scripts/exp301r_finalize.py" in text
```

Also assert no forbidden tuning flags (`--lr`, `--loops`, `--threshold`, `--sample-count`, `--task-weight`, `--bootstrap-seed`) occur.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_workflow.py -q`

Expected: missing workflow file.

- [ ] **Step 3: Implement workflow**

Use `actions/upload-artifact@v4` / `actions/download-artifact@v4`, write workflow SHA-256 into environment after checkout, keep run-id beacon stable across retries, use `fail-fast: false`, and make every downstream job depend only on complete upstream artifacts. Do not use Enterprise/larger/self-hosted runners.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_exp301r_workflow.py tests/test_exp301r_recovery.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/exp301r-standard-runner-sharded-recovery.yml tests/test_exp301r_workflow.py
git commit -m "ci(exp301r): add standard-runner sharded scientific recovery"
```

### Task 8: Exact-head verification, recovery freeze and dispatch registration

**Files:**
- Create: `protocols/v017/exp301r_execution_identity_v1.json`
- Create: `protocols/v017/exp301r_execution_identity_v1.sha256`
- Update recovery isolation allow-list to permit exactly these marker files.
- Register byte-identical workflow on default branch only after recovery branch exact-head CI is green.

**Interfaces:**
- Recovery identity binds original frozen implementation digest, prior failed run id, recovery source head/tree, recovery workflow digest, shard count 16, CPU device and schema.

- [ ] **Step 1: Extend tests RED**

Assert marker schema/digest reproduction, marker commit changes only the two recovery identity files after the implementation freeze, and exact workflow digest matches identity.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_exp301r_recovery.py tests/test_exp301r_workflow.py -q`

Expected: recovery marker missing.

- [ ] **Step 3: Freeze recovery implementation**

Generate canonical identity and sidecar from exact Git objects; commit only the two marker files.

- [ ] **Step 4: Verify complete test suite and existing EXP-301 contracts**

Run:

```bash
python scripts/verify_exp301_freeze.py
python scripts/verify_exp301r_recovery.py
pytest tests/test_exp301r_recovery.py tests/test_exp301r_workflow.py -q
pytest tests/test_exp301_runner.py tests/test_exp301_analysis.py tests/test_exp301_cross_root.py tests/test_exp301_evidence.py -q
pytest -q
```

Expected: all green.

- [ ] **Step 5: Register and dispatch**

Copy the exact recovery workflow blob to default branch for GitHub workflow registration without modifying the recovery scientific head. Dispatch only the frozen recovery branch. Record run id and recovery beacon on PR #75 / recovery PR provenance.

- [ ] **Step 6: Audit results before any scientific claim**

Require 24 trial artifacts, 4 selection artifacts, 4 challenge manifests, 64 prediction shards, 4 root evidences, 4 root recovery receipts, unchanged cross-root evidence and one final recovery envelope. Read the exact reducer decision from the final audited artifact; never infer it from job success.