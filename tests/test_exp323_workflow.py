from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/exp323-second-decay-convergence.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_scientific_workflow_is_manual_standard_runner_and_input_free() -> None:
    text = _text()
    assert "workflow_dispatch:" in text
    assert "inputs:" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "arm: [HOLD_5E5, DECAY_2P5E5]" in text
    assert "fail-fast: false" in text


def test_workflow_binds_authoritative_parent_and_source_checkpoint() -> None:
    text = _text()
    for token in (
        "35339004168",
        "10544985252",
        "18eb23aa09086aea89bf9ab96e07c52270069b75e8dc276f7610148b46b2e4cc",
        "10534351390",
        "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3",
        "exp319-stage-a-A_FIXED-0.0001-c3",
        "exp322-teacher-forced-intervention-35339004168",
    ):
        assert token in text


def test_workflow_has_no_scale_or_user_selected_scientific_controls() -> None:
    text = _text().lower()
    for token in (
        "--model-size",
        "--learning-rate",
        "--seed",
        "--checkpoint-step",
        "target_parameters: 30000000",
        "target_parameters: 100000000",
        "c_nrs_core",
    ):
        assert token not in text


def test_workflow_replays_parent_before_post2048_continuation() -> None:
    text = _text()
    assert "--parent-exp322 parent-exp322/exp322-final.json" in text
    assert "--execution-identity marker/protocols/v017/exp323_execution_identity_v1.json" in text
    assert "Run sealed EXP-323 replay plus continuation arm" in text


def test_reducer_waits_for_both_arms_and_uploads_one_final_evidence() -> None:
    text = _text()
    assert "needs: intervention" in text
    assert "exp323-HOLD_5E5-${{ github.run_id }}" in text
    assert "exp323-DECAY_2P5E5-${{ github.run_id }}" in text
    assert "exp323-final.json" in text
    assert "exp323-second-decay-convergence-${{ github.run_id }}" in text
