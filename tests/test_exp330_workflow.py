from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/".github/workflows/exp330-p02-conflict-projection-rescue.yml"
CON=ROOT/".github/workflows/v017-exp330-contract.yml"

def test_scientific_workflow_binds_exact_parent_and_reconstruction():
    text=SCI.read_text()
    for token in (
        "workflow_dispatch:",
        "exp330_execution_identity_v1.json",
        "35417293679",
        "10576262539",
        "d95a36e7305617b3b07232f18072ea99ef766a68161c6dbd9a707545253ae6a9",
        "f59682e2a08d3fd55437bfa3bfa6f4c3a88e950c4bc8a2ee7526774161be6e7f",
        "scripts/exp330_run.py",
    ):
        assert token in text

def test_scientific_workflow_contains_no_scale_or_architecture_knob():
    text=SCI.read_text().lower()
    for token in ("--learning-rate","--model-size","--d-model","--tokenizer","--layers"):
        assert token not in text

def test_contract_workflow_is_fast_pr_gate():
    text=CON.read_text()
    assert "pull_request:" in text
    for token in ("test_exp330_contract.py","test_exp330_runtime_surface.py","test_exp330_workflow.py"):
        assert token in text
