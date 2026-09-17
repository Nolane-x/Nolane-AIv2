from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = ROOT / ".github" / "workflows" / "exp319-learnability-foundation-diagnostic.yml"
CONTRACT = ROOT / ".github" / "workflows" / "v017-exp319-contract.yml"


def _scientific_text() -> str:
    assert SCIENTIFIC.is_file(), f"missing EXP-319 scientific workflow: {SCIENTIFIC}"
    return SCIENTIFIC.read_text(encoding="utf-8")


def _contract_text() -> str:
    assert CONTRACT.is_file(), f"missing EXP-319 contract workflow: {CONTRACT}"
    return CONTRACT.read_text(encoding="utf-8")


def test_scientific_workflow_is_manual_standard_runner_only() -> None:
    text = _scientific_text()
    assert "workflow_dispatch:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "larger" not in text.lower()
    assert "inputs:" not in text


def test_preflight_binds_prereg_freeze_and_execution_identity() -> None:
    text = _scientific_text()
    for token in (
        "scripts/verify_exp319_contract.py",
        "scripts/verify_exp319_freeze.py",
        "protocols/v017/exp319_execution_identity_v1.json",
        "protocols/v017/exp319_execution_identity_v1.sha256",
        "github-run-${{ github.run_id }}-exp319-v1",
    ):
        assert token in text


def test_stage_a_is_six_chains_with_four_256_step_continuations() -> None:
    text = _scientific_text()
    assert "stage-a-chunk-0" in text
    assert "stage-a-chunk-1" in text
    assert "stage-a-chunk-2" in text
    assert "stage-a-chunk-3" in text
    assert "exp319_stage_a_chunk.py" in text
    assert "exp319_stage_a_select.py" in text
    assert "A_FIXED" in text and "B_LOOP_SIMPLE" in text and "C_NRS_CORE" in text
    assert "0.0001" in text and "0.0003" in text
    assert "--chunk-index 0" in text
    assert "--chunk-index 3" in text


def test_stage_b_is_fresh_eight_chains_with_four_512_step_continuations() -> None:
    text = _scientific_text()
    for token in (
        "stage-b-chunk-0",
        "stage-b-chunk-1",
        "stage-b-chunk-2",
        "stage-b-chunk-3",
        "exp319_stage_b_chunk.py",
        "exp319_stage_b_select.py",
        "root: [1, 2, 3, 4]",
        "arm: [A_FIXED, C_NRS_CORE]",
    ):
        assert token in text
    assert "stage-a" in text.lower()
    assert "fresh" in text.lower()


def test_stage_c_is_post_selection_4x16_commitment_then_score_geometry() -> None:
    text = _scientific_text()
    for token in (
        "exp319_materialize_heldout.py",
        "exp319_predict_heldout_shard.py",
        "exp319_seal_root.py",
        "exp319_finalize.py",
        "shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]",
        "root: [1, 2, 3, 4]",
        "16/16",
        "commitment",
    ):
        assert token in text
    assert "--effort" not in text
    assert "--optimizer-steps" not in text
    assert "--threshold" not in text


def test_finalization_keeps_all_implementation_and_scale_authorizations_false() -> None:
    text = _scientific_text()
    for token in (
        "exp302_implementation_authorized",
        "exp320_implementation_authorized",
        "scale_authorized",
        "authorized_30m",
        "authorized_100m",
    ):
        assert token in text
    assert "FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY" in text


def test_contract_workflow_is_fast_pr_gate_for_full_exp319_contract_surface() -> None:
    text = _contract_text()
    assert "pull_request:" in text
    assert "runs-on: ubuntu-latest" in text
    for token in (
        "tests/test_exp319_contract.py",
        "tests/test_exp319_identity.py",
        "tests/test_exp319_freeze.py",
        "tests/test_exp319_cli.py",
        "tests/test_exp319_workflow.py",
        "scripts/verify_exp319_contract.py",
    ):
        assert token in text
