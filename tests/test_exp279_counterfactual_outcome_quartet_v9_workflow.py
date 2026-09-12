from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CONTRACT = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-ci.yml"
SCIENTIFIC = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-scientific.yml"
FREEZE_GUARD = WORKFLOWS / "exp279-counterfactual-outcome-quartet-v9-freeze-guard.yml"
SEAL_REPAIR = WORKFLOWS / "exp279-v9-cross-seal-repair.yml"
RELEASE_MARKER = "protocols/exp279-v9-counterfactual-outcome-quartet-release.lock"
SEAL_REPAIR_MARKER = "protocols/exp279-v9-cross-seal-repair.lock"
SCIENTIFIC_NAME = "exp279-counterfactual-outcome-quartet-v9-scientific"
SOURCE_RUN = "34695665034"
SOURCE_MERGE = "e7d08b0540dea1e7e7dff68667f7ad65108c4221"


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


def test_v9_cross_seal_repair_is_marker_only_and_immutable_source_only() -> None:
    text = SEAL_REPAIR.read_text(encoding="utf-8")
    lower = text.lower()
    assert "name: exp279-v9-cross-seal-repair" in text
    assert SEAL_REPAIR_MARKER in text
    assert RELEASE_MARKER not in text
    assert "workflow_dispatch" not in text
    assert "scientific-shard" not in text
    assert "strategy:" not in text
    assert "matrix:" not in text
    assert f"SOURCE_RUN_ID: {SOURCE_RUN}" in text
    assert f"SOURCE_EXECUTED_COMMIT: {SOURCE_MERGE}" in text
    assert "SOURCE_60_ARTIFACT_ID: 10299085674" in text
    assert "SOURCE_60_ZIP_SHA256: 8cc00c607315b1f34f97656f14edcc5d8fdcc6a92bf26240c90b3fdf0e9f1fed" in text
    assert "SOURCE_60_RECEIPT_SHA256: c4be74d32c263fae0f792a7e652d2dc55a1498457c4e6c2cae7b5834b014cabf" in text
    assert "SOURCE_120_ARTIFACT_ID: 10299080761" in text
    assert "SOURCE_120_ZIP_SHA256: 0ac9b76ee78c97c68eec8f90208490162c67dc880d0f13bf166531e32ee553e4" in text
    assert "SOURCE_120_RECEIPT_SHA256: 2e28e19d2e0def8710cf687b8a56fd78718e784c2384b26601f4e4afbb0dddd8" in text
    assert "exp279-v9-budget-60-e7d08b0540dea1e7e7dff68667f7ad65108c4221" in text
    assert "exp279-v9-budget-120-e7d08b0540dea1e7e7dff68667f7ad65108c4221" in text
    assert "run-id: ${{ env.SOURCE_RUN_ID }}" in text
    assert "github-token: ${{ github.token }}" in text
    assert "classify_exp279_counterfactual_outcome_quartet_v9_dev.py" in text
    assert 'assert data["decision"] == "QUARTET_MECHANISM_COURT_FAILED"' in text
    assert 'assert data["authorization_scope"] == "NONE"' in text
    assert 'assert data["successor_design_authorized"] is False' in text
    assert 'assert data["mechanism_successor_authorized"] is False' in text
    assert "validate_cross_receipt" in text
    assert "canonical_receipt_bytes" in text
    assert "receipt.json.sha256" in text
    assert "retention-days: 90" in text
    assert "60000" not in text
    assert "confirmatory" not in lower
    assert "challenge" not in lower
    assert "secrets." not in lower
