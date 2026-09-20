from pathlib import Path


def test_contract_workflow_covers_exp336_tests():
    text = Path(".github/workflows/v017-exp336-contract.yml").read_text()
    for name in (
        "test_exp336_contract.py",
        "test_exp336_identity.py",
        "test_exp336_runtime_surface.py",
        "test_exp336_preregistration.py",
        "test_exp336_workflow.py",
    ):
        assert name in text


def test_scientific_workflow_is_dispatch_only_and_eight_chunk_chain():
    text = Path(".github/workflows/exp336-cnrs-full32-stage-a.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "GITHUB_RUN_ATTEMPT" in text
    assert "artifact-digest" in text
    assert "10533822077" in text
    assert "10534076546" in text
    assert "10590026763" in text
    assert "10595335243" in text
    assert "exp336_execution_identity_v1.json" in text
    for index in range(8):
        assert f"chunk{index}:" in text
        assert f"--chunk-index {index}" in text
    assert "scripts/exp336_finalize.py" in text
    assert "stage_b_implementation_authorized" in text
    assert "authorized_100m" in text
