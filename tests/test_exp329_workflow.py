from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/".github/workflows/exp329-p02-cross-update-interference.yml"
CON=ROOT/".github/workflows/v017-exp329-contract.yml"

def test_scientific_workflow_binds_exact_parent_and_reconstruction():
    text=SCI.read_text()
    for token in (
        "workflow_dispatch:",
        "exp329_execution_identity_v1.json",
        "35414453731",
        "10575945812",
        "b15b7910abc91c4ba1b5fb3f64af3daa784fa400eae98983442c7565bf94e3d5",
        "e60f45b966954d48ad5d84ba65743925c87aa948fe15759f5860c88e9f6e77ba",
        "scripts/exp329_run.py",
    ):
        assert token in text

def test_scientific_workflow_contains_no_intervention_or_scale_knob():
    text=SCI.read_text().lower()
    for token in ("--learning-rate","--model-size","--d-model","--tokenizer","--gradient-surgery","--pcgrad"):
        assert token not in text

def test_contract_workflow_is_fast_pr_gate():
    text=CON.read_text()
    assert "pull_request:" in text
    for token in ("test_exp329_contract.py","test_exp329_runtime_surface.py","test_exp329_workflow.py"):
        assert token in text
