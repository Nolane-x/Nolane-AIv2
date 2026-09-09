from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp277-real-gate-b-ceremony.yml"
ARM_PATH = ".github/ceremony/exp277-real-gate-b-arm.json"
PROTOCOL_SHA = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def _workflow_text() -> str:
    assert WORKFLOW.is_file(), f"missing dormant real EXP-277 ceremony workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def test_exp277_real_gate_b_workflow_exists_and_is_arm_only() -> None:
    text = _workflow_text()
    assert "branches: [exp277-real-gate-b-ceremony]" in text
    assert ARM_PATH in text
    assert "workflow_dispatch" not in text
    assert "actions: read" in text
    assert "contents: read" in text
    assert "cancel-in-progress: false" in text


def test_exp277_real_gate_b_arm_commit_is_only_orchestration_marker() -> None:
    text = _workflow_text()
    assert "source_baseline_sha" in text
    assert "git rev-parse HEAD^" in text
    assert "git diff --name-only" in text
    assert ARM_PATH in text
    for path in (
        "src",
        "scripts",
        "protocols/stage_a_v1.json",
        "protocols/stage_a_v1.sha256",
    ):
        assert path in text
    assert PROTOCOL_SHA in text


def test_exp277_real_gate_b_requires_baseline_normal_and_focused_ci() -> None:
    text = _workflow_text()
    assert "actions/workflows/ci.yml/runs" in text
    assert "actions/workflows/exp277-confirmatory-ci.yml/runs" in text
    assert "head_sha=${BASELINE_SHA}" in text
    assert "event=pull_request" in text
    for job in ("core (3.11)", "core (3.13)", "model-smoke", "contract"):
        assert job in text


def test_exp277_real_gate_b_freezes_design_geometry_without_tiny_mode() -> None:
    text = _workflow_text()
    for token in (
        "--root-seed 20260906-exp277-paired-dev",
        "--d-model 64",
        "--hidden-size 48",
        "--target-parameters 500000",
        "--train-replicates 16",
        "--eval-replicates 32",
        "--eval-start-replicate 10000",
        "--batch-size 8",
        "--timesteps 4",
        "--variables 6",
        "--constraints 3",
        "--noise-std 0.05",
        "--lr 0.002",
        "--weight-decay 0.0",
    ):
        assert token in text
    assert "--tiny" not in text


def test_exp277_real_gate_a_precedes_future_public_beacon_and_uses_runtime_seal_time() -> None:
    text = _workflow_text()
    gate_a = text.index("python scripts/prepare_exp277_confirmatory_gate_a.py")
    beacon = text.index("https://api.drand.sh")
    assert gate_a < beacon
    assert "--checkpoint-seal-created-at-utc" not in text
    assert "checkpoint_seal_created_at_utc" in text
    assert "freeze_commit_timestamp_utc" in text
    assert "https://api2.drand.sh" in text
    assert "EXTERNAL_EVIDENCE_RECORDED" in text
    assert "cryptographic_signature_verified_by_ceremony" in text
    assert "False" in text or "false" in text


def test_exp277_real_gate_a_normalizes_git_commit_time_to_utc_before_sealing() -> None:
    text = _workflow_text()
    start = text.index("- name: Freeze real Gate A checkpoint and machinery before beacon")
    end = text.index("- name: Record first future public drand beacon after both freeze boundaries")
    block = text[start:end]
    assert "git show -s --format=%cI" in block
    assert "datetime.fromisoformat" in block
    assert "astimezone(timezone.utc)" in block
    assert "replace('+00:00', 'Z')" in block


def test_exp277_real_gate_b_writes_inference_marker_before_scientific_cli() -> None:
    text = _workflow_text()
    marker = "exp277-real-gate-b-inference-started-${{ github.sha }}"
    assert marker in text
    marker_index = text.index(marker)
    gate_b_index = text.index("python scripts/run_exp277_confirmatory_gate_b.py")
    assert marker_index < gate_b_index
    assert "actions/upload-artifact@v4" in text[marker_index:gate_b_index]
    assert "--arm-scientific-lane" in text[gate_b_index:]


def test_exp277_real_gate_b_preserves_all_outcomes_and_uploads_evidence_agnostically() -> None:
    text = _workflow_text()
    for token in (
        "NOT_READY_PRIMARY_BASELINE_NONPOSITIVE",
        "NOT_READY_VARIANCE_EXCEEDS_MAX_N",
        "INVALID_RUN",
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
        "wall_energy_per_episode",
        "confirmatory-raw.json",
        "confirmatory-analysis.json",
        "public-beacon-receipt.json",
        "public-beacon-external-evidence.json",
        "SHA256SUMS",
        "if: always()",
    ):
        assert token in text
    assert "assert analysis['decision'] == 'PROMOTE_TO_NEXT_STAGE'" not in text
    assert 'assert analysis["decision"] == "PROMOTE_TO_NEXT_STAGE"' not in text


def test_exp277_real_gate_b_rejects_replay_after_inference_or_outcome_artifact() -> None:
    text = _workflow_text()
    assert "actions/artifacts?name=" in text
    assert "exp277-real-gate-b-inference-started-${{ github.sha }}" in text
    assert "exp277-real-gate-b-outcome-${{ github.sha }}" in text
    assert "authoritative EXP-277" in text
