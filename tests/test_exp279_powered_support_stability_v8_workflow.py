from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CONTRACT = WORKFLOWS / "exp279-powered-support-stability-v8-ci.yml"
SCIENTIFIC = WORKFLOWS / "exp279-powered-support-stability-v8-scientific.yml"
FREEZE_GUARD = WORKFLOWS / "exp279-powered-support-stability-v8-freeze-guard.yml"
RELEASE_MARKER = "protocols/exp279-v8-powered-support-release.lock"


def test_v8_contract_workflow_is_engineering_only_and_cpu_bound() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "name: exp279-powered-support-stability-v8-ci" in text
    assert "download.pytorch.org/whl/cpu" in text
    for test_name in (
        "test_exp279_powered_support_stability_v8.py",
        "test_exp279_powered_support_stability_v8_cli.py",
        "test_exp279_powered_support_stability_v8_cross_cell.py",
        "test_exp279_powered_support_stability_v8_receipt_sidecars.py",
        "test_exp279_powered_support_stability_v8_workflow.py",
    ):
        assert test_name in text
    assert "scientific-shard" not in text
    assert RELEASE_MARKER not in text


def test_v8_scientific_workflow_is_one_time_marker_triggered_and_frozen() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    assert "name: exp279-powered-support-stability-v8-scientific" in text
    assert RELEASE_MARKER in text
    assert "train_replicates: [60, 120]" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "1332" in text
    assert "--scientific-branch-head ${{ github.event.pull_request.head.sha }}" in text
    assert "--executed-commit ${{ github.sha }}" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "receipt.json.sha256" in text
    assert "hashlib.sha256" in text
    assert "retention-days: 90" in text
    assert "60000" not in text
    assert "confirmatory" not in text.lower()
    assert "challenge" not in text.lower()


def test_v8_freeze_guard_can_cancel_later_scientific_reruns() -> None:
    text = FREEZE_GUARD.read_text(encoding="utf-8")
    assert "name: exp279-powered-support-stability-v8-freeze-guard" in text
    assert "actions: write" in text
    assert "exp279-powered-support-stability-v8-scientific" in text
    assert "authoritative" in text.lower()
    assert "/cancel" in text
    assert "queued" in text
    assert "in_progress" in text
