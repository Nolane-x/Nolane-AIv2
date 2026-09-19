from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / ".github/workflows/exp333-complete-eight-world-pair-lattice.yml"
CON = ROOT / ".github/workflows/v017-exp333-contract.yml"

def test_scientific_workflow_pins_parent_and_reconstruction_authority():
    text = SCI.read_text()
    for token in (
        "workflow_dispatch:",
        "exp333_execution_identity_v1.json",
        "10547681681",
        "10580076061",
        "35427969759",
        "02bf65b708e58837aa5b2a61c3af6acff09d000f8916bcda3a27a59ebdcd6355",
        "scripts/exp333_run.py",
    ):
        assert token in text

def test_scientific_workflow_exposes_no_training_or_scale_knob():
    text = SCI.read_text().lower()
    for token in ("--learning-rate","--model-size","--d-model","--tokenizer","--layers","--projection-threshold"):
        assert token not in text

def test_scientific_workflow_downloads_artifacts_by_id():
    text = SCI.read_text()
    assert 'artifact-ids: "10547681681"' in text
    assert 'artifact-ids: "10580076061"' in text
    assert 'run-id: "35345351869"' in text
    assert 'run-id: "35427969759"' in text
    assert "actions/download-artifact@v5" in text
    assert "github-token: ${{ github.token }}" in text

def test_contract_workflow_runs_exact_exp333_suite():
    text = CON.read_text()
    assert "pull_request:" in text
    for token in ("test_exp333_contract.py","test_exp333_runtime_surface.py","test_exp333_workflow.py"):
        assert token in text
