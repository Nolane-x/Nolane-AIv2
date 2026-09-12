from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CONTRACT = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-ci.yml"
SCIENTIFIC = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-scientific.yml"
FREEZE_GUARD = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-freeze-guard.yml"
RELEASE_MARKER = "protocols/exp279-v9-counterfactual-outcome-quartet-release.lock"
SCIENTIFIC_NAME = "exp279-counterfactual-outcome-quartet-v9-scientific"


def test_v9_contract_workflow_remains_engineering_only() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "name: exp279-counterfactual-outcome-quartet-v9-ci" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "tests/test_exp279_counterfactual_outcome_quartet_v9*.py" in text
    assert RELEASE_MARKER not in text
    assert "scientific-shard" not in text


def test_v9_scientific_workflow_is_marker_only_frozen_and_augmentation_only() -> None:
    text = SCIENTIFIC.read_text(encoding="utf-8")
    lower = text.lower()
    assert f"name: {SCIENTIFIC_NAME}" in text
    assert RELEASE_MARKER in text
    assert "workflow_dispatch" not in text
    assert "permissions:\n  contents: read" in text
    assert "actions: write" not in text
    assert "ref: ${{ github.event.pull_request.head.sha }}" in text
    assert 'LAST_TOUCH=$(git log -1 --format=%H -- "$MARKER")' in text
    assert 'test "$LAST_TOUCH" = "${{ github.event.pull_request.head.sha }}"' in text
    assert "EXP279_V9_COUNTERFACTUAL_OUTCOME_QUARTET_RELEASE_V1" in text
    for frozen in (
        "train_budgets=60,120",
        "canonical_indices=0,1,2,3",
        "fit_roots_per_canonical=2",
        "decision_roots_per_canonical=2",
        "fit_replicates_per_root=1332",
        "decision_replicates_per_root=1332",
        "student_family=OUTCOME_QUARTET_MLP",
        "student_hidden=64",
        "student_steps=200",
        "student_lr=0.001",
        "policy_route_rule=score_gt_0",
        "protocol_digest=c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440",
    ):
        assert frozen in text
    assert "train_replicates: [60, 120]" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "--scientific-branch-head ${{ github.event.pull_request.head.sha }}" in text
    assert "--executed-commit ${{ github.sha }}" in text
    assert "run_exp279_counterfactual_outcome_quartet_v9_dev.py" in text
    assert "aggregate_exp279_counterfactual_outcome_quartet_v9_dev.py" in text
    assert "classify_exp279_counterfactual_outcome_quartet_v9_dev.py" in text
    assert "receipt.json.sha256" in text
    assert "hashlib.sha256" in text
    assert "retention-days: 90" in text
    assert "60000" not in text
    assert "evaluation" not in lower
    assert "confirmatory" not in lower
    assert "challenge" not in lower
    assert "secrets." not in lower


def test_v9_freeze_guard_preserves_first_scientific_run_and_cancels_later_active_runs() -> None:
    text = FREEZE_GUARD.read_text(encoding="utf-8")
    lower = text.lower()
    assert "name: exp279-counterfactual-outcome-quartet-v9-freeze-guard" in text
    assert "contents: read" in text
    assert "actions: write" in text
    assert SCIENTIFIC_NAME in text
    assert "authoritative" in lower
    assert "/cancel" in text
    assert "queued" in text
    assert "in_progress" in text
    assert "pending" in text
    assert "workflow_dispatch" not in text
