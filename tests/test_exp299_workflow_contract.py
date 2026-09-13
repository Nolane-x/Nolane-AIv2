from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp299-native-fidelity-development.yml"


def test_development_workflow_is_marker_only_exact_head_and_four_root() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: exp299-native-fidelity-development" in text
    assert "protocols/exp299-native-fidelity-release.lock" in text
    assert "workflow_dispatch:" not in text
    assert "cancel-in-progress: false" in text
    assert "ref: ${{ github.event.pull_request.head.sha }}" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "Require this run to be the first authoritative DEVELOPMENT run" in text
    assert "Require marker-only frozen EXP-299 release at exact PR head" in text
    assert "Run frozen EXP-299 canonical root" in text
    assert "Verify root receipt and sidecar before upload" in text
    assert "Verify exactly four roots and reduce cross-root court" in text
    assert "Verify sealed cross receipt and boundaries" in text


def test_development_workflow_uses_cpu_model_runtime_and_portable_validation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in text
    assert "run_exp299_native_fidelity_dev.py root" in text
    assert "run_exp299_native_fidelity_dev.py cross" in text
    assert "validate_exp299_root" in text
    assert "validate_exp299_cross" in text
    assert "reconstruct=True" not in text
    assert "scientific_evidence_eligible" in text
    assert "exp300_execution_authorized" in text


def test_development_workflow_keeps_artifacts_for_independent_audit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/upload-artifact@v4" in text
    assert "actions/download-artifact@v4" in text
    assert "retention-days: 90" in text
    assert "execution-identity.json" in text
    assert "root-summary.json" in text
    assert "cross-summary.json" in text
