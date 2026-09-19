from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/".github/workflows/exp328-p02-order-geometry.yml"
CON=ROOT/".github/workflows/v017-exp328-contract.yml"

def test_scientific_workflow_is_manual_and_binds_exact_parent():
    text=SCI.read_text()
    for token in (
        "workflow_dispatch:",
        "exp328_execution_identity_v1.json",
        "35413434081",
        "10574837015",
        "b4ddab033a85762c99451f6532f790fbf4cda9181ff67c1a2093613dd0156d77",
        "27f8ef1fbe2bbb2a2a931d9dddd1409a6f3a0b9a8ecb06902f1e46bb07543279",
        "scripts/exp328_run.py",
    ):
        assert token in text

def test_workflow_contains_no_scale_or_architecture_intervention():
    text=SCI.read_text().lower()
    for token in ("--model-size","--d-model","--layers","--tokenizer","--learning-rate"):
        assert token not in text

def test_contract_workflow_is_fast_pr_gate():
    text=CON.read_text()
    assert "pull_request:" in text
    for token in ("test_exp328_contract.py","test_exp328_runtime_surface.py","test_exp328_workflow.py"):
        assert token in text
