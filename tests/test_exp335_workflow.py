from pathlib import Path


def test_contract_workflow_covers_exp335_tests():
    text = Path(".github/workflows/v017-exp335-contract.yml").read_text()
    for name in (
        "test_exp335_contract.py",
        "test_exp335_identity.py",
        "test_exp335_runtime_surface.py",
        "test_exp335_workflow.py",
    ):
        assert name in text


def test_scientific_workflow_is_dispatch_only_and_eight_chunk_chain():
    text = Path(".github/workflows/exp335-full32-afixed-foundation-reentry.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "GITHUB_RUN_ATTEMPT" in text
    assert "artifact-digest" in text
    assert "10547681681" in text
    assert "10583626513" in text
    assert "exp335_execution_identity_v1.json" in text
    for index in range(8):
        assert f"chunk{index}:" in text
        assert f"--chunk-index {index}" in text
    assert "scripts/exp335_finalize.py" in text
    assert "authorized_100m" in text
