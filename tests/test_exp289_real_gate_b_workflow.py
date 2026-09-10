from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp289-real-gate-b-ceremony.yml"


def test_exp289_real_gate_b_workflow_closes_irreversible_evidence_order() -> None:
    assert WORKFLOW.is_file(), "EXP-289 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "exp289-real-gate-b-ceremony",
        ".github/ceremony/exp289-real-gate-b-arm.json",
        "cancel-in-progress: false",
        "Validate arm-only commit and frozen scientific scope",
        "Reject replay after authoritative inference or outcome",
        "Require exact baseline normal and focused CI",
        "protocols/exp289_authoritative_development_v1.json",
        "seal_exp289_confirmatory_gate_a.py",
        "Record first future public drand beacon after both freeze boundaries",
        "target_round = math.floor(",
        "first_drand_round_strictly_after_actual_gate_a_seal",
        "primary_url = f'https://api.drand.sh/public/{target_round}'",
        "api2.drand.sh/public/",
        "seal_created_at_utc",
        "Publish irreversible inference-start replay barrier",
        "Execute frozen EXP-289 raw challenge once",
        "execute_exp289_confirmatory_challenge",
        "Publish immutable EXP-289 raw evidence before analysis",
        "Analyze persisted EXP-289 raw evidence once",
        "build_exp289_confirmatory_analysis",
        "Publish authoritative EXP-289 outcome marker",
        "Publish outcome-agnostic EXP-289 evidence bundle",
    )
    for item in required:
        assert item in text, f"missing real-ceremony invariant: {item}"

    forbidden = (
        "--test-only",
        "entropy_hex=\"ab\" * 32",
        "build_test_beacon_receipt",
        "api.drand.sh/public/latest",
    )
    for item in forbidden:
        assert item not in text, f"real ceremony contains forbidden or weaker path: {item}"

    inference_barrier = text.index("Publish irreversible inference-start replay barrier")
    raw_execution = text.index("Execute frozen EXP-289 raw challenge once")
    raw_persistence = text.index("Publish immutable EXP-289 raw evidence before analysis")
    analysis = text.index("Analyze persisted EXP-289 raw evidence once")
    outcome = text.index("Publish authoritative EXP-289 outcome marker")
    assert inference_barrier < raw_execution < raw_persistence < analysis < outcome


def test_exp289_raw_execution_requires_durable_inference_barrier() -> None:
    assert WORKFLOW.is_file(), "EXP-289 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("- name: Execute frozen EXP-289 raw challenge once")
    end = text.index("- name: Detect raw evidence", start)
    raw_step = text[start:end]
    assert "steps.barrier.outcome == 'success'" in raw_step, (
        "EXP-289 scientific raw execution must not start unless the irreversible "
        "inference-start artifact was durably uploaded"
    )


def test_exp289_real_gate_b_workflow_requires_arm_only_baseline_and_no_scientific_drift() -> None:
    assert WORKFLOW.is_file(), "EXP-289 real Gate-B ceremony workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "source_baseline_sha" in text
    assert "git diff --name-only" in text
    assert "src scripts protocols/stage_a_v1.json protocols/stage_a_v1.sha256" in text
    assert "event == 'push'" in text or "run.get('event') == 'push'" in text
    assert "core (3.11)" in text
    assert "core (3.13)" in text
    assert "model-smoke" in text
    assert "exp289-confirmatory-ci.yml" in text
    assert "confirmatory_data_consumed" in text
    assert "EV-E3" in text
