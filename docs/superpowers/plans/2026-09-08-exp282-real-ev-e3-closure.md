# EXP-282 Real EV-E3 Confirmatory-Open Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and durably persist the first real lineage-closed EXP-282 EV-E3 confirmatory-open result without modifying the frozen scientific implementation, while preventing post-result rerun selection.

**Architecture:** Keep the existing EXP-282 DEVELOPMENT, prep, seal, executor, and Analysis Court code unchanged. Add orchestration-only GitHub Actions plus static contract tests: the real ceremony is dormant until an explicit arming commit, waits for full normal CI success on its exact SHA, records an immutable pre-execution inference-start marker, executes at most one valid confirmatory attempt, and later persists only the exact completed Actions artifact without rerunning scientific code.

**Tech Stack:** GitHub Actions YAML, Python 3.11/3.13, PyTorch, stdlib JSON/hash handling, existing Nolane EXP-282 CLIs/validators, pytest, GitHub REST/`gh api`.

**Spec:** `docs/superpowers/specs/2026-09-08-exp282-real-ev-e3-closure-design.md`

## Global Constraints

- Repository: `Nolane-x/Nolane-AIv2`.
- Scientific baseline SHA: `77383b0a9c2ce92ded66f07234891790652a037b`.
- Frozen Stage-A V1 SHA-256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- Do not modify `src/`, `scripts/`, `protocols/stage_a_v1.json`, or `protocols/stage_a_v1.sha256` in this closure lane.
- Existing EXP-282 scientific modules/CLIs remain the sole scientific authority.
- Real ceremony geometry is fixed before the authoritative armed head: `d_model=64`, `hidden_size=48`, `target_parameters=500000`, `train_replicates=16`, `eval_replicates=32`, `eval_start_replicate=50000`, `batch_size=8`, `timesteps=6`, `variables=4`, `visibility_rate=0.5`, `noise_std=0.35`, `lr=0.002`, `weight_decay=0.0`.
- A valid `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, `PROMOTE_TO_NEXT_STAGE`, `HOLD_UNSTABLE`, or `KILL_SUBSYSTEM` is authoritative for its ceremony identity and cannot be replaced for preference.
- Once confirmatory inference starts, the same ceremony SHA must never execute scientific inference again.
- EV-E3 must keep `challenge_materialized=false`; EV-E4 100M post-freeze challenge replication is a separate future subproject.
- Ordinary CI must not execute real confirmatory observations.

---

### Task 1: Static ceremony workflow contract — RED first

**Files:**
- Create: `tests/test_exp282_real_ev_e3_workflows.py`
- Modify: `.github/workflows/ci.yml` only if explicit inclusion is required after confirming core pytest discovery does not already execute the test.

**Interfaces:**
- Consumes: repository files as text only; no scientific module invocation.
- Produces: a failing contract suite that defines the required ceremony/persistence workflow surfaces before those workflows exist.

- [ ] **Step 1: Create the implementation branch from the approved design/plan head**

Create branch `exp282-real-ev-e3-ceremony` from the exact commit containing this plan. Confirm compare against `main` contains only the approved spec/plan before adding tests.

- [ ] **Step 2: Write the failing static test**

Create `tests/test_exp282_real_ev_e3_workflows.py` with constants:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEREMONY = ROOT / ".github" / "workflows" / "exp282-real-ev-e3-ceremony.yml"
PERSISTENCE = ROOT / ".github" / "workflows" / "exp282-persist-real-ev-e3-evidence.yml"
BASELINE_SHA = "77383b0a9c2ce92ded66f07234891790652a037b"
PROTOCOL_SHA = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
```

The initial RED tests must require at least:

```python
def test_real_ev_e3_ceremony_workflow_exists():
    assert CEREMONY.is_file()


def test_ceremony_is_armed_only_by_explicit_marker_push():
    text = CEREMONY.read_text(encoding="utf-8")
    assert "exp282-real-ev-e3-ceremony" in text
    assert ".github/ceremony/exp282-real-ev-e3-arm.json" in text
    assert "workflow_dispatch" not in text


def test_ceremony_proves_frozen_scientific_identity():
    text = CEREMONY.read_text(encoding="utf-8")
    assert BASELINE_SHA in text
    assert "src scripts protocols/stage_a_v1.json protocols/stage_a_v1.sha256" in text
    assert "source_tree_digest" in text
    assert "scripts/verify_protocol.py" in text


def test_ceremony_freezes_exact_non_tiny_geometry():
    text = CEREMONY.read_text(encoding="utf-8")
    for token in (
        "--d-model 64", "--hidden-size 48", "--target-parameters 500000",
        "--train-replicates 16", "--eval-replicates 32",
        "--eval-start-replicate 50000", "--batch-size 8",
        "--timesteps 6", "--variables 4", "--visibility-rate 0.5",
        "--noise-std 0.35", "--lr 0.002", "--weight-decay 0.0",
    ):
        assert token in text
    assert "--tiny" not in text


def test_ceremony_waits_for_exact_head_ci_before_science():
    text = CEREMONY.read_text(encoding="utf-8")
    assert "GITHUB_SHA" in text
    assert "actions/workflows/ci.yml/runs" in text
    assert "event=pull_request" in text
    assert "conclusion" in text and "success" in text
    assert "model-smoke" in text


def test_ceremony_has_pre_execution_replay_barrier():
    text = CEREMONY.read_text(encoding="utf-8")
    marker = "exp282-real-ev-e3-inference-started-${{ github.sha }}"
    assert marker in text
    upload = text.index(marker)
    execute = text.index("python scripts/execute_exp282_confirmatory_ceremony.py")
    assert upload < execute
    assert "exp282-real-ev-e3-outcome-${{ github.sha }}" in text


def test_ceremony_does_not_hard_code_scientific_outcome():
    text = CEREMONY.read_text(encoding="utf-8")
    assert "assert analysis['decision'] == 'PROMOTE_TO_NEXT_STAGE'" not in text
    assert "assert analysis[\"decision\"] == \"PROMOTE_TO_NEXT_STAGE\"" not in text
    assert "public-beacon" not in text.lower()
    assert "drand" not in text.lower()
```

Also add persistence tests that initially fail because `PERSISTENCE` is absent and later require it not to invoke any of:

```python
FORBIDDEN_PERSISTENCE_EXECUTION = (
    "run_exp282_paired_dev.py",
    "prepare_exp282_confirmatory_open.py",
    "seal_exp282_confirmatory_ceremony.py",
    "execute_exp282_confirmatory_ceremony.py",
    "build_exp282_confirmatory_analysis",
)
```

- [ ] **Step 3: Run/observe clean RED**

Expected targeted command:

```bash
pytest -q tests/test_exp282_real_ev_e3_workflows.py
```

Expected failure at this stage: ceremony workflow file is missing. In GitHub-only execution, commit the RED test and obtain an exact-head CI failure whose relevant failure is the missing ceremony workflow, while existing scientific tests remain green.

- [ ] **Step 4: Commit the RED contract**

Commit message:

```text
test(exp282): freeze real EV-E3 workflow contract
```

---

### Task 2: Dormant real EV-E3 ceremony workflow — GREEN static contract

**Files:**
- Create: `.github/workflows/exp282-real-ev-e3-ceremony.yml`
- Modify: `tests/test_exp282_real_ev_e3_workflows.py` only to add missing structural assertions discovered during GREEN, never to weaken the RED contract.

**Interfaces:**
- Consumes: explicit arming marker `.github/ceremony/exp282-real-ev-e3-arm.json`; existing EXP-282 CLIs and validators; GitHub Actions metadata.
- Produces: one evidence artifact `exp282-real-ev-e3-evidence-${{ github.sha }}`, an optional inference-start marker, and one outcome marker for a valid attempt.

- [ ] **Step 1: Define a dormant trigger**

The workflow trigger must be exactly a branch/path push gate, not routine PR CI and not manual dispatch:

```yaml
name: exp282-real-ev-e3-ceremony

on:
  push:
    branches: [exp282-real-ev-e3-ceremony]
    paths:
      - .github/ceremony/exp282-real-ev-e3-arm.json

permissions:
  actions: read
  contents: read
```

The workflow file itself is therefore safe to add before the arming marker exists.

- [ ] **Step 2: Add source/arm preflight before model work**

Checkout with `fetch-depth: 0`. Parse the arm marker and require:

```json
{
  "schema": "NLM-EXP-282-REAL-EV-E3-ARM-V1",
  "ceremony_id": "EXP-282-REAL-EV-E3-2026-09-08",
  "source_baseline_sha": "77383b0a9c2ce92ded66f07234891790652a037b",
  "armed": true
}
```

Require branch `exp282-real-ev-e3-ceremony`. Fail if any scientific path differs from baseline:

```bash
BASELINE=77383b0a9c2ce92ded66f07234891790652a037b
test -z "$(git diff --name-only "$BASELINE" HEAD -- src scripts protocols/stage_a_v1.json protocols/stage_a_v1.sha256)"
```

- [ ] **Step 3: Add replay/outcome preflight**

Before installing model dependencies, query repository Actions artifacts with `gh api` for exact artifact names:

```text
exp282-real-ev-e3-inference-started-${GITHUB_SHA}
exp282-real-ev-e3-outcome-${GITHUB_SHA}
```

If either exists and is not expired, fail closed. This prevents scientific rerun of a SHA that already started inference and prevents rerunning a valid `NOT_READY` outcome.

- [ ] **Step 4: Wait for exact-head normal CI success**

Using `GH_TOKEN=${{ github.token }}`, poll:

```text
/repos/${GITHUB_REPOSITORY}/actions/workflows/ci.yml/runs?head_sha=${GITHUB_SHA}&event=pull_request&per_page=20
```

Select a run with exact `head_sha == GITHUB_SHA`. Wait until `status == completed`; require `conclusion == success`. Fetch `/actions/runs/<run_id>/jobs?per_page=100` and require successful jobs covering `core (3.11)`, `core (3.13)`, and `model-smoke`. Timeout/failure aborts before scientific work.

- [ ] **Step 5: Install frozen environment and capture source identity**

Run:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev,model]'
python scripts/verify_protocol.py
```

Write `/tmp/exp282-source-identity.json` from Python containing:

```python
{
    "schema": "NLM-EXP-282-REAL-EV-E3-SOURCE-IDENTITY-V1",
    "ceremony_sha": os.environ["GITHUB_SHA"],
    "scientific_baseline_sha": BASELINE_SHA,
    "protocol_digest": file_sha256(ROOT / "protocols/stage_a_v1.json"),
    "source_tree_digest": source_tree_digest(ROOT),
}
```

Require the protocol digest equals the canonical frozen Stage-A V1 digest.

- [ ] **Step 6: Build the fixed non-tiny DEVELOPMENT planning baseline**

Run exactly:

```bash
python scripts/run_exp282_paired_dev.py \
  --root-seed 20260906-exp282-paired-dev \
  --d-model 64 \
  --hidden-size 48 \
  --target-parameters 500000 \
  --train-replicates 16 \
  --eval-replicates 32 \
  --eval-start-replicate 50000 \
  --batch-size 8 \
  --timesteps 6 \
  --variables 4 \
  --visibility-rate 0.5 \
  --noise-std 0.35 \
  --lr 0.002 \
  --weight-decay 0.0 \
  --output /tmp/exp282-development.json \
  --registry-output /tmp/exp282-registry.json \
  --checkpoint-output /tmp/exp282-paired.pt
```

Immediately validate `validate_exp282_paired_development(...) == []` and `validate_neural_arm_registry(...) == []`; require artifact `code_digest == source_tree_digest(ROOT)` and checkpoint SHA matches its manifest.

- [ ] **Step 7: Freeze prep exactly once and branch on its authoritative state**

Run:

```bash
python scripts/prepare_exp282_confirmatory_open.py \
  --execution /tmp/exp282-development.json \
  --registry /tmp/exp282-registry.json \
  --output /tmp/exp282-prep.json
```

Validate prep. Accept only:

```text
CONFIRMATORY_OPEN_PREPARED
NOT_READY_VARIANCE_EXCEEDS_MAX_N
```

For `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, do not invoke seal or executor. Create summary with `confirmatory_data_consumed=false`, stage source/development/registry/checkpoint/prep/SHA256SUMS, upload the evidence artifact, then upload `exp282-real-ev-e3-outcome-${GITHUB_SHA}` containing the not-ready state. End the job successfully as a valid negative closure state.

- [ ] **Step 8: Seal a ready prep**

For `CONFIRMATORY_OPEN_PREPARED`, run:

```bash
python scripts/seal_exp282_confirmatory_ceremony.py \
  --execution /tmp/exp282-development.json \
  --prep /tmp/exp282-prep.json \
  --output /tmp/exp282-seal.json
```

Validate with `validate_exp282_confirmatory_ceremony_seal(...) == []`, and assert only structural frozen boundaries: EV-E2, UNVERIFIED, data false, seed materialization NOT_EXECUTED, challenge false, exact source/checkpoint/protocol lineage.

- [ ] **Step 9: Publish inference-start marker before executor invocation**

Create `/tmp/exp282-inference-started.json` with:

```python
{
    "schema": "NLM-EXP-282-REAL-EV-E3-INFERENCE-START-V1",
    "ceremony_sha": GITHUB_SHA,
    "source_tree_digest": source_tree_digest(ROOT),
    "protocol_digest": canonical_protocol_digest,
    "prep_digest": prep["prep_digest"],
    "ceremony_seal_digest": seal["ceremony_seal_digest"],
    "confirmatory_n": seal["confirmatory_n"],
    "reserved_replicate_ids": seal["reserved_replicate_ids"],
    "inference_started": True,
}
```

Upload it using `actions/upload-artifact@v4` with exact name `exp282-real-ev-e3-inference-started-${{ github.sha }}` and `if-no-files-found: error`. The executor step must appear only after this upload step.

- [ ] **Step 10: Execute the existing sealed ceremony exactly once**

Run:

```bash
python scripts/execute_exp282_confirmatory_ceremony.py \
  --seal /tmp/exp282-seal.json \
  --execution /tmp/exp282-development.json \
  --checkpoint /tmp/exp282-paired.pt \
  --raw-output /tmp/exp282-raw.json \
  --analysis-output /tmp/exp282-analysis.json \
  --bundle-output /tmp/exp282-result.json
```

Do not wrap this with alternate seeds, retry loops, geometry fallbacks, or decision-dependent reruns.

- [ ] **Step 11: Validate and stage the scientific result without re-deciding it**

Load outputs and require:

```python
validate_exp282_confirmatory_open_raw(raw) == []
validate_exp282_confirmatory_analysis(analysis) == []
validate_exp282_confirmatory_ceremony_result(result) == []
raw["evidence_level"] == "EV-E2"
raw["decision"] == "UNVERIFIED"
raw["confirmatory_data_consumed"] is True
raw["challenge_materialized"] is False
analysis["evidence_level"] == "EV-E3"
analysis["decision"] in {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
analysis["challenge_materialized"] is False
result["decision"] == analysis["decision"]
```

Create `ceremony-summary.json` by copying validated decision/effect/guard/digest fields; do not recompute a second decision rule.

Stage:

```text
development-planning-baseline.json
development-arm-registry.json
paired-functional-checkpoint.pt
confirmatory-prep.json
ceremony-seal.json
confirmatory-raw.json
confirmatory-analysis.json
ceremony-result.json
ceremony-summary.json
source-identity.json
SHA256SUMS
```

Upload as `exp282-real-ev-e3-evidence-${{ github.sha }}`. Then upload `exp282-real-ev-e3-outcome-${{ github.sha }}` with the exact validated decision and evidence artifact lineage.

- [ ] **Step 12: Run static tests to GREEN and commit**

Expected:

```bash
pytest -q tests/test_exp282_real_ev_e3_workflows.py
```

Commit message:

```text
ci(exp282): add dormant real EV-E3 ceremony
```

---

### Task 3: Draft PR and exact-head pre-arming verification

**Files:**
- No scientific files.
- PR metadata only.

**Interfaces:**
- Consumes: unarmed ceremony branch head.
- Produces: exact-head normal CI evidence proving the orchestration head is regression-clean before science is armed.

- [ ] **Step 1: Open a draft PR to `main`**

PR title:

```text
Run EXP-282 real EV-E3 confirmatory-open ceremony
```

Body must state that real science is still dormant because the arm marker is absent and list the frozen geometry/source/protocol boundaries.

- [ ] **Step 2: Audit changed-file scope**

Require no changes under:

```text
src/
scripts/
protocols/stage_a_v1.json
protocols/stage_a_v1.sha256
```

- [ ] **Step 3: Wait for exact-head normal CI 3/3 GREEN**

Use the PR head SHA returned by GitHub. Require completed/success jobs for core 3.11, core 3.13, and model-smoke. Do not arm while any job is queued/running/failed.

- [ ] **Step 4: Reviewer/self-review gate**

Inspect workflow patch and static tests for Important/Critical findings, especially hidden rerun paths, `--tiny`, outcome hard-coding, scientific source drift, or a missing pre-execution marker.

---

### Task 4: Arming commit and first-valid real ceremony

**Files:**
- Create: `.github/ceremony/exp282-real-ev-e3-arm.json`

**Interfaces:**
- Consumes: exact regression-clean unarmed ceremony workflow.
- Produces: one authoritative real ceremony attempt on the arm commit SHA.

- [ ] **Step 1: Create the arm marker as the only arming-commit change**

Content:

```json
{
  "schema": "NLM-EXP-282-REAL-EV-E3-ARM-V1",
  "ceremony_id": "EXP-282-REAL-EV-E3-2026-09-08",
  "source_baseline_sha": "77383b0a9c2ce92ded66f07234891790652a037b",
  "armed": true
}
```

Commit message:

```text
ceremony(exp282): arm first real EV-E3 attempt
```

- [ ] **Step 2: Hold branch head stable**

Do not push any other commit while both PR CI and ceremony run for the arm SHA are active. The ceremony itself waits for same-SHA PR CI success before scientific execution.

- [ ] **Step 3: Observe pre-inference phases**

Verify from job steps/logs that exact-head CI gate, source identity, DEVELOPMENT, and prep execute in order. If prep is `NOT_READY_VARIANCE_EXCEEDS_MAX_N`, treat that as the valid first outcome and proceed directly to persistence; do not create a new geometry commit.

- [ ] **Step 4: Observe inference barrier and execution when ready**

Before executor starts, confirm Actions has created `exp282-real-ev-e3-inference-started-<arm SHA>`. If later execution fails, the marker forbids a same-SHA rerun; preserve the failure and review before any new identity.

- [ ] **Step 5: Read the first valid outcome from the evidence artifact**

Do not infer decision from logs alone. Fetch the exact evidence artifact metadata and inspect `ceremony-summary.json` plus underlying validated scientific artifact(s). Record:

```text
ceremony run id
ceremony head SHA
artifact id/name/digest
source tree digest
protocol digest
prep status
confirmatory n (if ready)
seal digest (if ready)
raw digest (if executed)
analysis digest (if executed)
decision / not-ready state
primary effect and protected guards (if executed)
challenge_materialized=false
```

---

### Task 5: Persistence-only workflow pinned to the first valid artifact

**Files:**
- Create: `.github/workflows/exp282-persist-real-ev-e3-evidence.yml`
- Modify: `tests/test_exp282_real_ev_e3_workflows.py`

**Interfaces:**
- Consumes: exact first-valid ceremony run/artifact IDs discovered in Task 4.
- Produces: immutable files under `evidence/exp282/2026-09-08-real-ev-e3-confirmatory-open/` without invoking scientific execution.

- [ ] **Step 1: Extend the static contract before creating persistence workflow**

Require the persistence workflow to contain pinned literals for the exact ceremony run ID, artifact ID, artifact digest, and arm SHA. Require it to contain `SHA256SUMS`, Actions run/artifact metadata validation, the evidence destination path, and frozen-source diff checks. Require every token in `FORBIDDEN_PERSISTENCE_EXECUTION` to be absent from the workflow text.

Run targeted test and observe RED because persistence workflow does not yet exist.

- [ ] **Step 2: Implement the persistence-only workflow**

Trigger only on pushes to `exp282-real-ev-e3-ceremony` whose changed path is this persistence workflow. Permissions:

```yaml
permissions:
  actions: read
  contents: write
```

It must:

1. checkout with sufficient history;
2. prove no diff vs `77383b0a...` under `src scripts protocols/stage_a_v1.json protocols/stage_a_v1.sha256`;
3. fetch the exact pinned artifact metadata via `gh api` and require exact ID/name/digest/run/head SHA;
4. fetch the exact pinned run metadata and require completed/success;
5. download only the exact evidence artifact;
6. verify every original `SHA256SUMS` entry;
7. for `NOT_READY`, validate prep and assert data was never consumed;
8. for executed EV-E3, invoke only existing `validate_*` functions on persisted JSON and require lineage consistency, without invoking builders/executors and without requiring a preferred scientific decision;
9. write `PROVENANCE.md`, `actions-run-metadata.json`, and `actions-artifact-metadata.json`;
10. commit only `evidence/exp282/2026-09-08-real-ev-e3-confirmatory-open/` to the ceremony branch.

- [ ] **Step 3: Run static workflow tests to GREEN**

```bash
pytest -q tests/test_exp282_real_ev_e3_workflows.py
```

Commit message:

```text
ci(exp282): pin real EV-E3 evidence persistence
```

- [ ] **Step 4: Observe persistence run**

Require completed/success. If persistence fails, fix/retry only persistence mechanics; do not rerun the ceremony or regenerate scientific artifacts.

---

### Task 6: Evidence-only integration review and exact-head CI

**Files:**
- Evidence files generated by Task 5.
- PR metadata/body.

**Interfaces:**
- Consumes: persisted first-valid evidence.
- Produces: merge-ready evidence-only PR head with exact-head full CI.

- [ ] **Step 1: Audit final changed files**

Allowed categories:

```text
.github/workflows/exp282-real-ev-e3-ceremony.yml
.github/workflows/exp282-persist-real-ev-e3-evidence.yml
.github/ceremony/exp282-real-ev-e3-arm.json
tests/test_exp282_real_ev_e3_workflows.py
docs/superpowers/specs/2026-09-08-exp282-real-ev-e3-closure-design.md
docs/superpowers/plans/2026-09-08-exp282-real-ev-e3-closure.md
evidence/exp282/2026-09-08-real-ev-e3-confirmatory-open/**
```

Reject any `src/`, `scripts/`, or frozen protocol change.

- [ ] **Step 2: Update PR body with factual first-valid outcome**

Include source SHA/digest, protocol digest, run/artifact IDs, prep status, n, evidence level/decision if executed, effect/guard values copied from validated artifacts, challenge boundary, and EV-E4/EV-E5 blockers. Do not broaden claims.

- [ ] **Step 3: Run fresh exact-head normal CI**

Require core 3.11, core 3.13, model-smoke all completed/success on the exact final evidence head.

- [ ] **Step 4: Review PR comments/reviews/mergeability/head drift**

No unresolved Important/Critical review issue. `mergeable=true`. Exact head still equals the verified CI SHA.

---

### Task 7: Squash merge, post-merge verification, and scientific fork

**Files:**
- No source edits during merge.

**Interfaces:**
- Consumes: exact verified PR head.
- Produces: durable EXP-282 EV-E3 closure on `main` and a next-step decision consistent with the frozen result.

- [ ] **Step 1: Squash merge with expected-head protection**

Pass the exact verified head SHA to the merge action so any drift rejects the merge.

- [ ] **Step 2: Verify `main` identity**

Fetch `main` independently and require it points to the returned squash SHA.

- [ ] **Step 3: Verify post-merge CI**

Require post-merge push CI 3/3 success on that exact squash SHA, including model-smoke and protocol verification.

- [ ] **Step 4: Record closure evidence**

Add a PR closure comment containing merge SHA, exact-head pre-merge CI, post-merge CI, evidence directory, first-valid ceremony run/artifact IDs, and the bounded scientific result.

- [ ] **Step 5: Follow the frozen decision fork**

- `PROMOTE_TO_NEXT_STAGE`: start a new brainstorming/design subproject for **EXP-282 EV-E4 integrated 100M post-freeze public-beacon challenge replication**. Do not implement EV-E4 before that design is approved.
- `HOLD_UNSTABLE`: persist/stop; no same-protocol outcome-shopping rerun.
- `KILL_SUBSYSTEM`: persist/stop; do not run EV-E4 for the explicit-belief hypothesis.
- `NOT_READY_VARIANCE_EXCEEDS_MAX_N`: persist/stop; any future redesign is a new development/protocol decision.

## Plan Self-Review

- Spec coverage: scientific-source immutability, fixed geometry, sample-size fail-closed state, exact-head CI before science, first-valid-attempt authority, inference-start replay barrier, immutable evidence, persistence-only semantics, decision fork, EV-E3 challenge boundary, and post-merge verification are each mapped to explicit tasks.
- Placeholder scan: no TODO/TBD, "similar to", or unspecified validation steps remain.
- Type/name consistency: artifact names, branch names, workflow paths, schemas, baseline SHA, and ceremony IDs are consistent across tasks.
- Scope check: this plan closes only EXP-282 real EV-E3. EV-E4, EV-E5, and other Stage-A gates remain separate architectural subprojects.
