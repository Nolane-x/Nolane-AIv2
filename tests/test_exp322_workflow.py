from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/exp322-teacher-forced-learnability.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_scientific_workflow_is_manual_standard_runner_and_input_free() -> None:
    text = _text()
    assert "workflow_dispatch:" in text
    assert "inputs:" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "arm: [HOLD_1E4, DECAY_5E5]" in text
    assert "fail-fast: false" in text


def test_workflow_binds_parent_checkpoint_and_exp321_evidence() -> None:
    text = _text()
    for token in (
        "35331243762",
        "5f3bd6cbacef0af4883e33f14bafd56b49a307df88641e837bb9ecdf004d5091",
        "10534351390",
        "18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb",
        "exp319-stage-a-A_FIXED-0.0001-c3",
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


def test_reducer_waits_for_both_arms_and_uploads_one_final_evidence() -> None:
    text = _text()
    assert "needs: intervention" in text
    assert "exp322-HOLD_1E4-${{ github.run_id }}" in text
    assert "exp322-DECAY_5E5-${{ github.run_id }}" in text
    assert "exp322-final.json" in text
    assert "exp322-teacher-forced-intervention-${{ github.run_id }}" in text


def test_workflow_verifies_artifact_digests_and_embeds_execution_identity() -> None:
    text = _text()
    assert "10541366011" in text
    assert "491f4fe0207a3ebbaba7122bbe254cf9bed44339e14aeb4a40b6a23928aecbe9" in text
    assert "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3" in text
    assert "--execution-identity marker/protocols/v017/exp322_execution_identity_v1.json" in text
