# EXP-289 Confirmatory Gate-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed EXP-289 Stage-A confirmatory path from DEVELOPMENT RDER evidence through immutable pre-beacon authority, while preserving the frozen protocol and EV-E2/UNVERIFIED boundary until a valid real Gate B.

**Architecture:** Keep EXP-289-specific statistics separate from EXP-286 log-cost machinery. Gate A derives paired relative RDER reductions only from reconstruction-valid DEVELOPMENT rows, freezes sample size before challenge randomness, and refuses zero/invalid baseline denominators. Later authority layers bind the exact trained state, code tree, analysis contract, challenge contract, and post-freeze beacon without allowing CI rehearsal to promote evidence.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, GitHub Actions, canonical SHA-256 evidence contracts.

**Spec:** `docs/superpowers/specs/2026-09-10-exp289-confirmatory-gate-a-design.md`

## Global Constraints

- Frozen protocol file and digest must not change: `protocols/stage_a_v1.json` / canonical digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- EXP-289 primary endpoint remains `repeat_dead_end_rate`, lower is better.
- MESI remains 0.25 relative reduction.
- Confirmatory paired n remains 32..128; power target remains 0.90; familywise alpha remains 0.05.
- Protected over-prune ceiling remains 0.005; protected solution floor remains `local_nogood >= no_nogood - 0.01`.
- Zero-opportunity episodes remain in raw evidence/safety accounting but are excluded from the primary RDER denominator.
- No epsilon denominator rescue.
- DEVELOPMENT observations are never reclassified as confirmatory observations.
- No real beacon/challenge materialization before immutable Gate-A authority is complete.
- TEST-ONLY ceremony paths remain EV-E2 / UNVERIFIED.

---

### Task 1: RDER-specific Gate-A preparation

**Files:**
- Create: `src/nolane_ai/experiments/exp289_confirmatory_prep.py`
- Create: `tests/test_exp289_confirmatory_prep.py`

**Interfaces:**
- Consumes: `validate_exp289_paired_artifact(payload)` or the existing EXP-289 DEVELOPMENT validator, neural arm registry, canonical frozen Stage-A protocol.
- Produces: `frozen_exp289_gate_a_contract() -> dict`, `prepare_exp289_confirmatory_gate_a(...) -> dict`, `validate_exp289_confirmatory_prep(payload) -> list[str]`.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_frozen_exp289_gate_a_contract_matches_stage_a_v1():
    contract = frozen_exp289_gate_a_contract()
    assert contract["primary_endpoint"] == "repeat_dead_end_rate"
    assert contract["mesi_relative_reduction"] == 0.25
    assert contract["min_n"] == 32 and contract["max_n"] == 128
    assert contract["power_target"] == 0.90
    assert contract["familywise_alpha"] == 0.05
    assert contract["zero_opportunity_policy"] == "retain_raw_episode_exclude_from_rder_denominator"
    assert contract["epsilon_denominator_rescue"] is False


def test_gate_a_refuses_zero_baseline_rder_denominator(valid_execution, registry, protocol):
    for row in valid_execution["evaluation"]["per_replicate"]:
        row["no_nogood"]["repeat_dead_end_rate"] = 0.0
    valid_execution["artifact_digest"] = recompute_exp289_artifact_digest(valid_execution)
    with pytest.raises(ValueError, match="baseline RDER denominator"):
        prepare_exp289_confirmatory_gate_a(valid_execution, registry, protocol)


def test_gate_a_never_materializes_challenge(valid_execution_32, registry, protocol):
    prep = prepare_exp289_confirmatory_gate_a(valid_execution_32, registry, protocol)
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["confirmatory_data_consumed"] is False
    assert prep["challenge_materialized"] is False
    assert prep["confirmatory_lineage"]["seed_materialization_status"] == "NOT_EXECUTED"
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `pytest -q tests/test_exp289_confirmatory_prep.py`

Expected: collection/import failure because `nolane_ai.experiments.exp289_confirmatory_prep` does not exist.

- [ ] **Step 3: Implement minimal Gate-A statistics**

```python
EFFECT_TYPE = "paired_relative_rder_reduction"
MESI_RELATIVE_REDUCTION = 0.25
MIN_N, MAX_N, POWER_TARGET, FAMILYWISE_ALPHA = 32, 128, 0.90, 0.05


def _paired_effects(rows):
    effects = []
    for row in rows:
        baseline = row["no_nogood"]["repeat_dead_end_rate"]
        candidate = row["local_nogood"]["repeat_dead_end_rate"]
        if baseline is None or candidate is None or not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
            raise ValueError("EXP-289 pilot RDER must be finite and defined")
        if float(baseline) <= 0.0:
            raise ValueError("EXP-289 baseline RDER denominator must be strictly positive; no epsilon rescue")
        effects.append((float(baseline) - float(candidate)) / float(baseline))
    return effects


def _required_n(paired_sd):
    if paired_sd == 0.0:
        return 1
    z_alpha = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA)
    z_power = NormalDist().inv_cdf(POWER_TARGET)
    return max(1, math.ceil(((z_alpha + z_power) * paired_sd / MESI_RELATIVE_REDUCTION) ** 2))
```

The prep artifact must return `NOT_READY_VARIANCE_EXCEEDS_MAX_N` with no reserved IDs when required n >128, otherwise `CONFIRMATORY_GATE_A_PREPARED` with `confirmatory_n=max(32, required_n)` and disjoint reserved replicate IDs.

- [ ] **Step 4: Run Gate-A tests GREEN**

Run: `pytest -q tests/test_exp289_confirmatory_prep.py tests/test_exp289_paired_runner.py tests/test_exp289_ground_truth_boundary.py`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nolane_ai/experiments/exp289_confirmatory_prep.py tests/test_exp289_confirmatory_prep.py
git commit -m "feat(exp289): freeze confirmatory RDER Gate A"
```

---

### Task 2: Transactional Gate-A CLI and focused CI

**Files:**
- Create: `scripts/prepare_exp289_confirmatory_gate_a.py`
- Create: `tests/test_exp289_confirmatory_prep_cli.py`
- Create: `.github/workflows/exp289-confirmatory-ci.yml`

**Interfaces:**
- Consumes: Task 1 prep functions and canonical protocol verifier.
- Produces: no-overwrite JSON prep publication and an EXP-289-focused workflow.

- [ ] **Step 1: Write RED tests**

```python
def test_cli_refuses_existing_output(tmp_path, valid_inputs):
    out = tmp_path / "prep.json"
    out.write_text("sentinel")
    result = run_cli(valid_inputs, out)
    assert result.returncode != 0
    assert out.read_text() == "sentinel"


def test_workflow_runs_exp289_gate_a_contract():
    text = Path(".github/workflows/exp289-confirmatory-ci.yml").read_text()
    assert "test_exp289_confirmatory_prep.py" in text
    assert "prepare_exp289_confirmatory_gate_a.py" in text
```

- [ ] **Step 2: Observe RED**

Run: `pytest -q tests/test_exp289_confirmatory_prep_cli.py`

Expected: missing CLI/workflow failures only.

- [ ] **Step 3: Implement atomic publication and workflow**

```python
fd, staged = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(prep, handle, sort_keys=True, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.link(staged, output)
finally:
    Path(staged).unlink(missing_ok=True)
```

Focused CI must install `[dev,model]`, verify frozen protocol, run existing EXP-289 DEVELOPMENT tests, run Gate-A tests, execute a DEVELOPMENT-only 32-replicate tiny rehearsal, run prep, assert no challenge/confirmatory consumption, and compile `src scripts`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_exp289_confirmatory_prep.py tests/test_exp289_confirmatory_prep_cli.py tests/test_exp289_paired_runner.py tests/test_exp289_ground_truth_boundary.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_exp289_confirmatory_gate_a.py tests/test_exp289_confirmatory_prep_cli.py .github/workflows/exp289-confirmatory-ci.yml
git commit -m "ci(exp289): exercise confirmatory Gate A"
```

---

### Task 3: Authoritative DEVELOPMENT geometry and Gate-A evidence

**Files:**
- Create: `protocols/exp289_authoritative_development_v1.json`
- Create: `tests/test_exp289_authoritative_development.py`
- Modify: `.github/workflows/exp289-confirmatory-ci.yml`

**Interfaces:**
- Produces a canonical geometry digest and DEVELOPMENT execution/prep artifacts; never modifies Stage-A V1.

- [ ] **Step 1: RED-test exact geometry binding**

```python
def test_authoritative_geometry_is_not_tiny_smoke():
    cfg = load_geometry()
    assert cfg["experiment_id"] == "EXP-289"
    assert cfg["eval_replicates"] >= 32
    assert cfg["purpose"] == "AUTHORITATIVE_DEVELOPMENT_GATE_A"
```

- [ ] **Step 2: Observe RED**, then add a fixed JSON geometry using the already-supported runner parameters and a distinct DEVELOPMENT replicate range.

- [ ] **Step 3: Workflow executes exactly that JSON geometry**, writes its canonical SHA-256 into execution/prep lineage, and uploads only EV-E2 artifacts.

```python
assert execution["evidence_level"] == "EV-E2"
assert prep["confirmatory_data_consumed"] is False
assert prep["confirmatory_lineage"]["seed_materialization_status"] == "NOT_EXECUTED"
```

- [ ] **Step 4: Verify exact-head focused + normal CI and commit.**

---

### Task 4: Exact trained-state checkpoint court

**Files:**
- Modify: `src/nolane_ai/experiments/exp289_paired_runner.py`
- Create: `src/nolane_ai/experiments/exp289_checkpoint.py`
- Create: `tests/test_exp289_checkpoint.py`

**Interfaces:**
- DEVELOPMENT artifact freezes optimizer contract, per-step training losses, and final functional digests.
- `build_exp289_checkpoint(execution) -> dict` replays training from sealed lineage and refuses any loss/state mismatch.

- [ ] **Step 1: RED tests require optimizer/loss/final-state identity.**

```python
def test_checkpoint_replays_exact_trained_state(execution):
    checkpoint = build_exp289_checkpoint(execution)
    assert checkpoint["final_state_digests"] == execution["training"]["final_state_digests"]
    assert validate_exp289_checkpoint(checkpoint, execution) == []
```

- [ ] **Step 2: Observe RED.**
- [ ] **Step 3: Add exact replay + canonical checkpoint digest and fail-closed validator.**
- [ ] **Step 4: Run checkpoint + existing EXP-289 suites GREEN.**
- [ ] **Step 5: Commit.**

---

### Task 5: Execution court and immutable pre-beacon seal

**Files:**
- Create: `src/nolane_ai/experiments/exp289_challenge_worlds.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_authorization.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_ceremony.py`
- Create: `tests/test_exp289_confirmatory_authorization.py`
- Create: `tests/test_exp289_confirmatory_ceremony.py`

**Interfaces:**
- Authorization binds protocol digest, code-tree digest, authoritative geometry digest, DEVELOPMENT artifact digest, prep digest, checkpoint digest, challenge-contract digest and frozen confirmatory n.
- Seal binds freeze commit SHA/timestamp and explicitly contains no beacon entropy, challenge seed or confirmatory observations.

- [ ] **Step 1: RED tamper tests** for each binding.
- [ ] **Step 2: Observe RED.**
- [ ] **Step 3: Implement canonical authorization + seal digests.**

```python
if prep["confirmatory_ready"] is not True:
    raise ValueError("EXP-289 execution authorization requires Gate A READY")
if checkpoint_errors:
    raise ValueError("EXP-289 checkpoint authority is not closed")
```

- [ ] **Step 4: Verify GREEN and commit.**

---

### Task 6: TEST-ONLY post-freeze challenge and Gate-B firewall

**Files:**
- Create: `src/nolane_ai/experiments/exp289_beacon.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_executor.py`
- Create: `src/nolane_ai/experiments/exp289_confirmatory_analysis.py`
- Create: tests for beacon, executor, analysis, and TEST-ONLY non-promotion.

**Interfaces:**
- Challenge seed = domain-separated SHA-256 over frozen protocol identity, freeze lineage, beacon receipt, experiment id, stream and replicate.
- No raw challenge-seed CLI override.
- Raw challenge artifact is persisted/validated before analysis.

- [ ] **Step 1: RED tests prove pre-freeze/replayed/tampered beacons fail and TEST-ONLY cannot promote.**
- [ ] **Step 2: Observe RED.**
- [ ] **Step 3: Implement generator/executor/analysis.**

```python
if beacon["mode"] == "TEST_ONLY":
    assert result["evidence_level"] == "EV-E2"
    assert result["decision"] == "UNVERIFIED"
    assert result["scientific_evidence_eligible"] is False
```

Real analysis uses paired bootstrap of relative RDER reduction, reports zero-opportunity handling explicitly, reconstructs over-prune safety from raw receipts, applies the frozen solution floor, and keeps integrity failures as `INVALID_RUN`.

- [ ] **Step 4: Verify focused TEST-ONLY ceremony GREEN and commit.**

---

### Task 7: Exact-head closure and real-ceremony eligibility audit

**Files:**
- Modify: PR body/comments only unless verification exposes a real defect.

**Interfaces:**
- Produces an exact-SHA engineering closure statement and a scientific boundary statement.

- [ ] **Step 1:** Lock PR HEAD SHA and run/fetch focused EXP-289 CI, normal CI core 3.11/3.13/model-smoke, and frozen-protocol verification.
- [ ] **Step 2:** Audit diff from `main@8c0b5add4387cbf244d5df61b2ef1a6ebac79a66`; assert no changes to `protocols/stage_a_v1.json` or `.sha256`.
- [ ] **Step 3:** If Gate A is NOT_READY, persist that EV-E2 disposition and do not build a real beacon ceremony. If READY, freeze exact source/config/evaluator/analysis identity before any future public beacon.
- [ ] **Step 4:** Invoke `superpowers:verification-before-completion` and `superpowers:requesting-code-review` before any merge-ready claim.
