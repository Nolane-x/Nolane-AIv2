from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp286-real-gate-b-ceremony.yml"


def test_exp286_real_gate_b_workflow_closes_irreversible_evidence_order() -> None:
    assert WORKFLOW.is_file(), "EXP-286 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "exp286-real-gate-b-ceremony",
        ".github/ceremony/exp286-real-gate-b-arm.json",
        "cancel-in-progress: false",
        "Validate arm-only commit and frozen scientific scope",
        "Reject replay after authoritative inference or outcome",
        "Publish irreversible EXP-286 ceremony arm lock",
        "Require exact baseline normal and focused CI",
        "seal_exp286_confirmatory_gate_a.py",
        "Record first future public drand beacon after both freeze boundaries",
        "first_drand_round_strictly_after_actual_gate_a_seal",
        "Publish irreversible inference-start replay barrier",
        "Execute frozen EXP-286 raw challenge once",
        "execute_exp286_confirmatory_challenge",
        "Publish immutable EXP-286 raw evidence before analysis",
        "Analyze persisted EXP-286 raw evidence once",
        "build_exp286_confirmatory_analysis",
        "Publish authoritative EXP-286 outcome marker",
        "Publish outcome-agnostic EXP-286 evidence bundle",
    )
    for item in required:
        assert item in text, f"missing real-ceremony invariant: {item}"

    forbidden = (
        "--test-only",
        "build_test_beacon_receipt",
        "api.drand.sh/public/latest",
    )
    for item in forbidden:
        assert item not in text, f"real ceremony contains forbidden or weaker path: {item}"

    arm_lock = text.index("Publish irreversible EXP-286 ceremony arm lock")
    beacon = text.index("Record first future public drand beacon after both freeze boundaries")
    barrier = text.index("Publish irreversible inference-start replay barrier")
    raw = text.index("Execute frozen EXP-286 raw challenge once")
    raw_persist = text.index("Publish immutable EXP-286 raw evidence before analysis")
    analysis = text.index("Analyze persisted EXP-286 raw evidence once")
    outcome = text.index("Publish authoritative EXP-286 outcome marker")
    assert arm_lock < beacon < barrier < raw < raw_persist < analysis < outcome


def test_exp286_raw_execution_requires_durable_inference_barrier() -> None:
    assert WORKFLOW.is_file(), "EXP-286 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("- name: Execute frozen EXP-286 raw challenge once")
    end = text.index("- name: Detect raw evidence", start)
    raw_step = text[start:end]
    assert "steps.barrier.outcome == 'success'" in raw_step


def test_exp286_real_gate_b_requires_arm_only_baseline_without_scientific_drift() -> None:
    assert WORKFLOW.is_file(), "EXP-286 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "source_baseline_sha",
        "git diff --name-only",
        "src scripts protocols/stage_a_v1.json protocols/stage_a_v1.sha256",
        "core (3.11)",
        "core (3.13)",
        "model-smoke",
        "exp286-confirmatory-ci.yml",
        "confirmatory_data_consumed",
        "EV-E3",
    )
    for item in required:
        assert item in text, f"missing baseline/evidence invariant: {item}"


def test_exp286_real_gate_b_propagates_actual_seal_time_into_validation_and_execution() -> None:
    assert WORKFLOW.is_file(), "EXP-286 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "seal_created_at_utc=actual_gate_a_seal_created_at_utc" in text
    assert "seal_created_at_utc=Path('/tmp/exp286-actual-gate-a-seal-created-at.txt').read_text().strip()" in text
    assert "receipt['receipt_digest']=_receipt_digest(receipt)" in text
