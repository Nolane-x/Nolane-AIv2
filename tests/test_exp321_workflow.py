from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = ROOT / ".github" / "workflows" / "exp321-afixed-failure-localization.yml"
CONTRACT = ROOT / ".github" / "workflows" / "v017-exp321-contract.yml"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_scientific_workflow_is_manual_standard_runner_and_input_free() -> None:
    text = _text(SCIENTIFIC)
    assert "workflow_dispatch:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "inputs:" not in text
    assert "fetch-depth: 0" in text


def test_scientific_workflow_binds_marker_prereg_and_exact_checkpoint() -> None:
    text = _text(SCIENTIFIC)
    for token in (
        "verify_exp321_freeze.py",
        "verify_exp321_contract.py",
        "protocols/v017/exp321_execution_identity_v1.json",
        "protocols/v017/exp321_execution_identity_v1.sha256",
        "10534351390",
        "exp319-stage-a-A_FIXED-0.0001-c3",
        "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3",
        "35311529822",
        "scripts/exp321_localize_checkpoint.py",
        "--execution-identity",
    ):
        assert token in text


def test_scientific_workflow_contains_no_training_or_scale_surface() -> None:
    text = _text(SCIENTIFIC)
    forbidden = (
        "optimizer.step",
        "run_training_chunk",
        "scientific_train_step",
        "--learning-rate",
        "run_scientific_trial",
        "stage-b-chunk",
    )
    for token in forbidden:
        assert token not in text.lower()


def test_scientific_workflow_fail_closes_authorization_and_reproduction() -> None:
    text = _text(SCIENTIFIC)
    assert 'p.get("reproduction_valid") is not True' in text
    for token in (
        "exp302_implementation_authorized",
        "exp320_implementation_authorized",
        "scale_authorized",
        "authorized_30m",
        "authorized_100m",
    ):
        assert token in text
    assert 'len(p.get("records", [])) != 128' in text


def test_contract_workflow_is_fast_pr_gate() -> None:
    text = _text(CONTRACT)
    assert "pull_request:" in text
    assert "runs-on: ubuntu-latest" in text
    for token in (
        "verify_exp321_contract.py",
        "test_exp321_contract.py",
        "test_exp321_measurement_surface.py",
        "test_exp321_identity.py",
        "test_exp321_freeze.py",
        "test_exp321_workflow.py",
    ):
        assert token in text
