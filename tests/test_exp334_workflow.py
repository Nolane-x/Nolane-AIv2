from pathlib import Path

def test_scientific_workflow_is_dispatch_only_and_pinned():
    text=Path(".github/workflows/exp334-higher-order-conflict-subspace-rescue.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    for artifact in ("10547681681","10581761422","10578071604","10574301567"):
        assert artifact in text
    assert "exp334_execution_identity_v1.json" in text
    assert "Fail closed" in text
    assert "authorized_100m" in text

def test_contract_workflow_covers_exp334_tests():
    text=Path(".github/workflows/v017-exp334-contract.yml").read_text()
    assert "test_exp334_contract.py" in text
    assert "test_exp334_runtime_surface.py" in text
    assert "test_exp334_workflow.py" in text
