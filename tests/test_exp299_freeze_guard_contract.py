from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp299-native-fidelity-freeze-guard.yml"


def test_freeze_guard_audits_first_authoritative_run_and_blocks_duplicates() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: exp299-native-fidelity-freeze-guard" in text
    assert "exp299-native-fidelity-development" in text
    assert "pull_request:" in text
    assert "actions: read" in text
    assert "authoritative_development_run_id" in text
    assert "duplicate_development_run_ids" in text
    assert "duplicate DEVELOPMENT runs detected" in text


def test_freeze_guard_has_no_execution_path_to_scientific_runner() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "run_exp299_native_fidelity_dev.py" not in text
    assert "workflow_dispatch:" not in text
    assert "upload-artifact" not in text
