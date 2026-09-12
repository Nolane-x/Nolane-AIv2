from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    classify_quartet_budget,
    classify_quartet_cross_budget,
    heldout_policy_metrics,
)


def _economic_root() -> dict[str, object]:
    stop = torch.tensor([True, True, True, True, False, False, False, False])
    branch = torch.tensor([True, True, False, True, True, True, False, False])
    route = torch.tensor([False, False, False, False, True, True, False, False])
    return heldout_policy_metrics(
        route,
        stop,
        branch,
        stop_accounted_flops=100,
        branch_accounted_flops=150,
        student_accounted_flops=1,
    )


def _collapsed_root() -> dict[str, object]:
    stop = torch.tensor([True, True, True, True, False, False, False, False])
    branch = torch.tensor([True, True, False, True, True, True, False, False])
    return heldout_policy_metrics(
        torch.zeros(8, dtype=torch.bool),
        stop,
        branch,
        stop_accounted_flops=100,
        branch_accounted_flops=150,
        student_accounted_flops=1,
    )


def test_v9_budget_requires_all_four_canonical_roots() -> None:
    passed = classify_quartet_budget([_economic_root() for _ in range(4)])
    assert passed["classification"] == "QUARTET_POLICY_RECURRENTLY_ECONOMIC"
    assert passed["root_classifications"] == ["QUARTET_ROOT_ECONOMIC"] * 4
    assert passed["pooled_policy_utility"] > passed["pooled_stop_baseline_utility"]
    assert passed["pooled_policy_utility"] > passed["pooled_branch_baseline_utility"]

    partial_roots = [_economic_root() for _ in range(3)] + [_collapsed_root()]
    partial = classify_quartet_budget(partial_roots)
    assert partial["classification"] == "QUARTET_POLICY_PARTIAL"


def test_v9_cross_budget_recomputes_and_only_passes_two_recurrent_budgets() -> None:
    roots = [_economic_root() for _ in range(4)]
    train60 = {"train_replicates": 60, "root_metrics": copy.deepcopy(roots), "classification": "TAMPERED"}
    train120 = {"train_replicates": 120, "root_metrics": copy.deepcopy(roots), "classification": "TAMPERED"}
    decision = classify_quartet_cross_budget(train60, train120)
    assert decision["decision"] == "QUARTET_MECHANISM_COURT_PASSED"
    assert decision["successor_design_authorized"] is True
    assert decision["mechanism_successor_authorized"] is True
    assert decision["authorization_scope"] == "IMPLEMENT_QUARTET_SUCCESSOR_DEVELOPMENT_ONLY"
    assert decision["scientific_evidence_eligible"] is False
    assert decision["fresh_evaluation_lineage_consumed"] is False
    assert decision["confirmatory_data_consumed"] is False
    assert decision["challenge_materialized"] is False
    assert decision["promotion_claimed"] is False

    train120["root_metrics"][-1] = _collapsed_root()
    failed = classify_quartet_cross_budget(train60, train120)
    assert failed["decision"] != "QUARTET_MECHANISM_COURT_PASSED"
    assert failed["successor_design_authorized"] is False
    assert failed["mechanism_successor_authorized"] is False
    assert failed["authorization_scope"] == "NONE"
