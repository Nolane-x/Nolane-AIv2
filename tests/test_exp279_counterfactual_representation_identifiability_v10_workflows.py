from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

CONTRACT = WORKFLOWS / "exp279-counterfactual-representation-identifiability-v10-contract.yml"
SCIENTIFIC = WORKFLOWS / "exp279-counterfactual-representation-identifiability-v10-scientific.yml"
FREEZE = WORKFLOWS / "exp279-counterfactual-representation-identifiability-v10-freeze-guard.yml"
MARKER = "protocols/exp279-v10-cric-release.lock"


def _text(path: Path) -> str:
    assert path.exists(), f"missing V10 workflow: {path}"
    return path.read_text(encoding="utf-8")


def test_v10_dedicated_contract_workflow_covers_frozen_surface() -> None:
    text = _text(CONTRACT)
    assert "exp279-counterfactual-representation-identifiability-v10-contract" in text
    for path_fragment in (
        "exp279_counterfactual_representation_identifiability_v10.py",
        "exp279_counterfactual_representation_identifiability_v10_runner.py",
        "exp279_counterfactual_representation_identifiability_v10_receipts.py",
        "test_exp279_counterfactual_representation_identifiability_v10",
        "scripts/verify_protocol.py",
        "python -m compileall -q src scripts",
    ):
        assert path_fragment in text
    assert "workflow_dispatch" not in text


def test_v10_scientific_workflow_is_marker_only_and_frozen() -> None:
    text = _text(SCIENTIFIC)
    assert "exp279-counterfactual-representation-identifiability-v10-scientific" in text
    assert MARKER in text
    assert "workflow_dispatch" not in text
    assert "contents: read" in text
    assert "LAST_TOUCH" in text
    assert "github.event.pull_request.head.sha" in text
    assert "train_replicates: [60, 120]" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "fit_replicates_per_root=128" in text
    assert "heldout_replicates_per_root=128" in text
    assert "views=V9_MEAN_BASELINE,FULL_PREBRANCH_STATE,EXECUTION_FRONTIER_STATE" in text
    assert "nearest_neighbor=k1_exact_squared_euclidean_fit_zscore" in text
    assert "query_chunk_size=256" in text
    assert "run_exp279_counterfactual_representation_identifiability_v10_dev.py" in text
    assert "aggregate_exp279_counterfactual_representation_identifiability_v10_dev.py" in text
    assert "classify_exp279_counterfactual_representation_identifiability_v10_dev.py" in text
    assert "retention-days: 90" in text
    assert "evaluation_rng_used" in text
    assert "confirmatory_data_consumed" in text
    assert "mechanism_successor_authorized" in text
    for forbidden in (
        "evaluate_stage_a_pilot.py",
        "execute_exp282_confirmatory_ceremony.py",
        "authorize_exp286_confirmatory_execution.py",
    ):
        assert forbidden not in text


def test_v10_freeze_guard_is_pr_scoped_first_run_authoritative() -> None:
    text = _text(FREEZE)
    assert "exp279-counterfactual-representation-identifiability-v10-freeze-guard" in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "github.event.pull_request.number" in text
    assert "exp279-counterfactual-representation-identifiability-v10-scientific" in text
    assert "NLM-EXP-279-V10-CRIC-FREEZE-GUARD-V1" in text
    assert "authoritative_scientific_run_id" in text
    assert "observed_scientific_run_ids" in text
    assert "cancelled_duplicate_run_ids" in text
    assert "/cancel" in text
