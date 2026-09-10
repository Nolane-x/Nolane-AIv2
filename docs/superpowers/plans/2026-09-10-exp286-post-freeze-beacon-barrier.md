# EXP-286 Post-Freeze Beacon Barrier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest fail-closed EXP-286 post-freeze public-beacon boundary needed before any confirmatory challenge entropy can exist.

**Architecture:** Keep Gate A and `AUTHORIZED_NOT_EXECUTED` unchanged. Add an experiment-local beacon receipt validator and deterministic challenge-seed derivation function that accepts only an externally supplied receipt, enforces publication strictly after the frozen authority timestamp when provided, distinguishes synthetic test receipts from scientific receipts, and never generates a beacon internally. No challenge worlds, confirmatory observations, or decision rule are executed in this task.

**Tech Stack:** Python 3.11/3.13, stdlib `datetime`/`hashlib`, canonical evidence hashing, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-exp286-oracle-conflict-headroom-design.md` plus frozen `protocols/stage_a_v1.json` EXP-286 authority.

## Global Constraints

- Experiment is exactly `EXP-286` / `H-CONFLICT-01`.
- Frozen challenge lane remains `POST_FREEZE_CHALLENGE`; challenge entropy comes only from a future public beacon after freeze.
- Primary endpoint remains `accounted_reasoning_flops_to_verified_solution`, direction lower.
- MESI remains relative reduction `0.15`; paired `min_n=32`, `max_n=128`, power `0.90`.
- DEVELOPMENT pilot observations are never reclassified as post-freeze challenge evidence.
- This task must not materialize challenge worlds, consume confirmatory data, execute the decision rule, or edit `protocols/stage_a_v1.json` / `.sha256`.

---

### Task 1: Public-beacon receipt and seed-derivation barrier

**Files:**
- Create: `tests/test_exp286_beacon.py`
- Create: `src/nolane_ai/experiments/exp286_beacon.py`
- Modify: `.github/workflows/exp286-confirmatory-ci.yml`

**Interfaces:**
- Produces: `build_test_beacon_receipt(...) -> dict[str, Any]`
- Produces: `validate_exp286_beacon_receipt(payload, *, freeze_commit_timestamp_utc=None, authorization_created_at_utc=None) -> list[str]`
- Produces: `derive_exp286_challenge_seed(*, protocol_digest, beacon_receipt, stream, replicate, freeze_commit_timestamp_utc=None, authorization_created_at_utc=None) -> int`
- Consumes: `nolane_ai.protocol.evidence.canonical_sha256`

- [ ] **Step 1: Write the failing tests**

```python
from copy import deepcopy
import pytest


def _api():
    try:
        from nolane_ai.experiments.exp286_beacon import (
            build_test_beacon_receipt,
            derive_exp286_challenge_seed,
            validate_exp286_beacon_receipt,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 post-freeze beacon barrier is missing")
    return build_test_beacon_receipt, derive_exp286_challenge_seed, validate_exp286_beacon_receipt


def test_test_beacon_is_deterministic_but_never_scientific():
    build, derive, validate = _api()
    receipt = build(
        source="test-fixture",
        beacon_id="fixture-1",
        published_at_utc="2026-09-10T05:00:00Z",
        entropy_hex="ab" * 32,
        evidence_reference="synthetic://fixture-1",
    )
    assert receipt["experiment_id"] == "EXP-286"
    assert receipt["test_only"] is True
    assert receipt["scientific_evidence_eligible"] is False
    assert validate(receipt) == []
    first = derive(protocol_digest="protocol-digest", beacon_receipt=receipt, stream="challenge", replicate=30032)
    second = derive(protocol_digest="protocol-digest", beacon_receipt=receipt, stream="challenge", replicate=30032)
    assert first == second


def test_real_beacon_must_be_strictly_post_freeze_and_post_authorization():
    _, _, validate = _api()
    payload = {
        "schema": "NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-286",
        "source": "external-public-source",
        "beacon_id": "public-1",
        "published_at_utc": "2026-09-10T05:00:00Z",
        "entropy_hex": "cd" * 32,
        "evidence_reference": "external-evidence-record",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "EXTERNAL_EVIDENCE_RECORDED",
        "receipt_digest": "",
    }
    from nolane_ai.protocol.evidence import canonical_sha256
    clean = deepcopy(payload)
    clean.pop("receipt_digest")
    payload["receipt_digest"] = canonical_sha256(clean)
    errors = validate(
        payload,
        freeze_commit_timestamp_utc="2026-09-10T05:00:00Z",
        authorization_created_at_utc="2026-09-10T04:59:59Z",
    )
    assert any("strictly after Gate A freeze" in item for item in errors)


def test_seed_derivation_rejects_tampered_receipt():
    build, derive, _ = _api()
    receipt = build(
        source="test-fixture",
        beacon_id="fixture-2",
        published_at_utc="2026-09-10T05:00:00Z",
        entropy_hex="ef" * 32,
        evidence_reference="synthetic://fixture-2",
    )
    receipt["entropy_hex"] = "00" * 32
    with pytest.raises(ValueError, match="invalid EXP-286 beacon receipt"):
        derive(protocol_digest="protocol-digest", beacon_receipt=receipt, stream="challenge", replicate=30032)
```

- [ ] **Step 2: Run RED verification**

Run: `pytest -q tests/test_exp286_beacon.py`

Expected: a clean failing test stating `EXP-286 post-freeze beacon barrier is missing`; no production module exists yet.

- [ ] **Step 3: Implement the minimal barrier**

Create `exp286_beacon.py` with schema `NLM-EXP-286-PUBLIC-BEACON-RECEIPT-V1`, at least 256 bits of hex entropy, UTC timestamp parsing, synthetic-vs-real evidence classification, canonical self-hash verification, strict post-freeze/post-authorization timestamp guards, and seed material:

```python
material = f"{protocol_digest}|{beacon_receipt['receipt_digest']}|EXP-286|{stream}|{replicate}"
seed = int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")
```

The module must not fetch a beacon, generate external entropy, materialize challenge data, or expose a function that upgrades evidence level.

- [ ] **Step 4: Run focused GREEN verification**

Run: `pytest -q tests/test_exp286_beacon.py tests/test_exp286_confirmatory_authorization.py tests/test_exp286_confirmatory_prep.py`

Expected: all pass.

- [ ] **Step 5: Put the new test under focused CI authority**

Add `tests/test_exp286_beacon.py` to the existing `pytest -q tests/test_exp286_*.py tests/test_matched_conflict_arms.py` coverage indirectly through the glob; add an explicit CI-contract assertion only if the existing workflow-contract test does not already prove the glob. Do not add beacon materialization or real entropy to CI.

- [ ] **Step 6: Exact-head verification**

Run/fetch both `exp286-confirmatory-ci` and normal `ci` for the resulting exact SHA. Required result before claiming engineering closure: focused PASS, core 3.11 PASS, core 3.13 PASS, model-smoke PASS, frozen protocol verification PASS.

- [ ] **Step 7: Commit boundaries**

Use one RED commit containing only the failing test, then one GREEN commit containing the minimal production module. Workflow-only hardening, if necessary, is a separate commit with its own contract test.
