from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / ".github" / "workflows" / "v017-exp301-contract.yml"
SCIENTIFIC = ROOT / ".github" / "workflows" / "exp301-scientific-court.yml"


def test_normal_ci_runs_only_test_only_exp301_smoke() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "tests/test_exp301_analysis.py" in text
    assert "tests/test_exp301_runner.py" in text
    assert "tests/test_exp301_workflows.py" in text
    assert "python scripts/run_exp301.py" in text
    assert "--test-only" in text
    assert "--execution-identity" not in text
    assert "scientific court" not in text.lower()


def test_scientific_workflow_is_manual_only_and_freeze_gated() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "schedule:" not in text
    assert "protocols/v017/exp301_execution_identity_v1.json" in text
    assert "protocols/v017/exp301_execution_identity_v1.sha256" in text
    assert "--execution-identity" in text
    assert "--test-only" not in text


def test_scientific_workflow_freezes_all_four_roots_and_no_tuning_surface() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    compact = "".join(text.split())
    assert "root:[0,1,2,3]" in compact
    assert "fail-fast:false" in compact
    assert "cross-root-reducer" in text
    assert "root-${{ matrix.root }}" in text
    for forbidden in ("--lr", "--loops", "--threshold", "--bootstrap-seed", "--task-weight", "--sample-count"):
        assert forbidden not in text


def test_scientific_workflow_uses_write_once_artifact_names_and_concurrency_guard() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text
    assert "exp301-scientific-${{ github.ref }}" in text
    assert "if-no-files-found: error" in text
    assert "overwrite" not in text.lower()
