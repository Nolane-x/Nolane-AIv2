from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SCI=ROOT/".github/workflows/exp331-pair-lattice-projection-safety.yml";CON=ROOT/".github/workflows/v017-exp331-contract.yml"

def test_scientific_workflow_binds_both_parents_and_reconstruction():
    text=SCI.read_text()
    for token in ("workflow_dispatch:","exp331_execution_identity_v1.json","35419828278","10577706134","cadf92521948973bb319e8280934e559526616097c75f74a1fc1a90d02c33ab7","35413434081","10574837015","b4ddab033a85762c99451f6532f790fbf4cda9181ff67c1a2093613dd0156d77","scripts/exp331_run.py"):
        assert token in text

def test_workflow_exposes_no_scale_or_training_hyperparameter_knob():
    text=SCI.read_text().lower()
    for token in ("--learning-rate","--model-size","--d-model","--tokenizer","--layers","--projection-threshold"):
        assert token not in text

def test_contract_workflow_runs_exact_suite():
    text=CON.read_text();assert "pull_request:" in text
    for token in ("test_exp331_contract.py","test_exp331_runtime_surface.py","test_exp331_workflow.py"):
        assert token in text
