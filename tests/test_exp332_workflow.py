from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / ".github/workflows/exp332-portable-pair-lattice-confirmation.yml"
CON = ROOT / ".github/workflows/v017-exp332-contract.yml"

def test_scientific_workflow_binds_both_exp327_witnesses_and_exp330():
    text = SCI.read_text()
    for token in (
        "workflow_dispatch:",
        "exp332_execution_identity_v1.json",
        "10574837015",
        "10578071604",
        "10577706134",
        "27f8ef1fbe2bbb2a2a931d9dddd1409a6f3a0b9a8ecb06902f1e46bb07543279",
        "cc608361850a524f6e82c06c8f65a5a7c0cb321a90b2c6d4686238a82233540f",
        "scripts/exp332_run.py",
    ):
        assert token in text

def test_scientific_workflow_exposes_no_training_or_scale_knob():
    text = SCI.read_text().lower()
    for token in ("--learning-rate", "--model-size", "--d-model", "--tokenizer", "--layers", "--projection-threshold"):
        assert token not in text

def test_contract_workflow_runs_exact_suite():
    text = CON.read_text()
    assert "pull_request:" in text
    for token in ("test_exp332_contract.py", "test_exp332_runtime_surface.py", "test_exp332_workflow.py"):
        assert token in text


def test_scientific_workflow_downloads_immutable_artifacts_by_id_v5():
    text = SCI.read_text()
    assert "actions/download-artifact@v5" in text
    for artifact_id in ("10547681681", "10577706134", "10574837015", "10578071604"):
        assert f'artifact-ids: "{artifact_id}"' in text
    for run_id in ("35345351869", "35419828278", "35413434081"):
        assert f'run-id: "{run_id}"' in text
    assert "github-token: ${{ github.token }}" in text
    assert "/actions/artifacts/{artifact_id}/zip" not in text
    assert "sha2556sum" not in text
    assert "sha256sum exp327-replay/exp327-final.json" in text
