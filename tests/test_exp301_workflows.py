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
    assert "--frozen-implementation-identity" not in text
    assert "scientific court" not in text.lower()


def test_scientific_workflow_is_manual_only_and_freeze_gated() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "schedule:" not in text
    assert "protocols/v017/exp301_execution_identity_v1.json" in text
    assert "protocols/v017/exp301_execution_identity_v1.sha256" in text
    assert "--frozen-implementation-identity" in text
    assert "--execution-identity" not in text
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


def test_scientific_root_jobs_bind_post_freeze_beacon_device_and_single_root_output() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "EXP301_DEVICE:" in text
    assert "EXP301_CHALLENGE_BEACON:" in text
    assert "github.run_id" in text
    assert "github.run_attempt" in text
    assert "--output-dir artifacts" in text
    assert "--output-dir artifacts/root-${{ matrix.root }}" not in text


def test_cross_root_reducer_is_live_and_uploads_write_once_evidence() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "scripts/reduce_exp301.py" in text
    assert "cross-root-evidence.json" in text
    assert "Freeze-gated reducer boundary" not in text
    assert "exp301-cross-root" in text
    assert "if-no-files-found: error" in text


def test_scientific_workflow_uses_write_once_artifact_names_and_concurrency_guard() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text
    assert "exp301-scientific-${{ github.ref }}" in text
    assert "if-no-files-found: error" in text
    assert "overwrite" not in text.lower()
