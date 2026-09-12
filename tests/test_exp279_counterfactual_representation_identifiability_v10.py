from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_representation_identifiability_v10 import (
    CROSS_BUDGET_IDENTIFIABLE,
    CROSS_BUDGET_NOT_IDENTIFIABLE,
    EXECUTION_FRONTIER_STATE,
    FULL_PREBRANCH_STATE,
    REPRESENTATION_DIMENSIONS,
    REPRESENTATION_ECONOMICALLY_IDENTIFIABLE,
    REPRESENTATION_NOT_IDENTIFIABLE,
    REPRESENTATION_PARTIAL,
    REPRESENTATION_RECURRENTLY_IDENTIFIABLE,
    V9_MEAN_BASELINE,
    classify_cross_budget,
    classify_representation_budget,
    classify_representation_root,
    counterfactual_action_labels,
    direct_policy_metrics,
    exact_1nn_predict,
    extract_representation_views,
    fit_standardizer,
)
from nolane_ai.experiments.matched_routing_arms import build_matched_exp279_arm_triplet


def _arm_and_batch(batch: int = 2):
    _, _, arm = build_matched_exp279_arm_triplet(
        d_model=64,
        hidden_size=48,
        target_parameters=500000,
        route_threshold=0.5,
        device="cpu",
    )
    surface = torch.randn(batch, 4, 64)
    variables = torch.randn(batch, 6, 64)
    incidence = torch.zeros(batch, 3, 6)
    incidence[:, 0, :2] = 1
    incidence[:, 1, 2:4] = 1
    incidence[:, 2, 4:] = 1
    return arm, surface, variables, incidence


def test_representation_views_have_frozen_dimensions_and_never_execute_branch(monkeypatch):
    arm, surface, variables, incidence = _arm_and_batch()

    def forbidden_branch(*args, **kwargs):
        raise AssertionError("branch preview is forbidden during V10 feature extraction")

    monkeypatch.setattr(arm, "_branch_context", forbidden_branch)
    views = extract_representation_views(
        arm,
        surface_events=surface,
        variable_states=variables,
        incidence=incidence,
    )

    assert list(views) == [V9_MEAN_BASELINE, FULL_PREBRANCH_STATE, EXECUTION_FRONTIER_STATE]
    assert REPRESENTATION_DIMENSIONS == {
        V9_MEAN_BASELINE: 144,
        FULL_PREBRANCH_STATE: 480,
        EXECUTION_FRONTIER_STATE: 312,
    }
    assert views[V9_MEAN_BASELINE].shape == (2, 144)
    assert views[FULL_PREBRANCH_STATE].shape == (2, 480)
    assert views[EXECUTION_FRONTIER_STATE].shape == (2, 312)
    assert "targets" not in inspect.signature(extract_representation_views).parameters


def _logits_from_predictions(predictions: torch.Tensor) -> torch.Tensor:
    logits = torch.full((*predictions.shape, 2), -5.0)
    logits.scatter_(-1, predictions.unsqueeze(-1), 5.0)
    return logits


def test_counterfactual_labels_partition_rescue_harm_both_success_both_failure():
    targets = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])
    stop_pred = torch.tensor([[1, 0], [0, 0], [0, 0], [1, 0]])
    branch_pred = torch.tensor([[0, 0], [1, 0], [0, 0], [1, 0]])

    labels, outcomes = counterfactual_action_labels(
        _logits_from_predictions(stop_pred),
        _logits_from_predictions(branch_pred),
        targets,
    )

    assert labels.tolist() == [1, 0, 0, 0]
    assert outcomes.tolist() == [0, 1, 2, 3]  # RESCUE, HARM, BOTH_SUCCESS, BOTH_FAILURE


def test_standardizer_is_fit_only_and_clamps_constant_dimensions():
    fit = torch.tensor([[1.0, 4.0], [3.0, 4.0]])
    mean, std = fit_standardizer(fit)
    assert torch.allclose(mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(std[0], torch.tensor(1.0))
    assert std[1].item() == pytest.approx(1e-6)


def test_exact_1nn_is_chunk_invariant_and_first_index_tie_breaking():
    fit_x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    fit_y = torch.tensor([1, 0, 0, 1], dtype=torch.int64)
    query = torch.tensor([[0.0, 0.0], [9.0, 9.0], [19.0, 19.0], [0.0, 0.0]])

    pred1 = exact_1nn_predict(fit_x, fit_y, query, chunk_size=1)
    pred3 = exact_1nn_predict(fit_x, fit_y, query, chunk_size=3)
    pred256 = exact_1nn_predict(fit_x, fit_y, query, chunk_size=256)

    assert pred1.tolist() == [1, 0, 1, 1]
    assert torch.equal(pred1, pred3)
    assert torch.equal(pred1, pred256)


def _economic_metrics(*, rescues: int, harms: int, route_count: int, episodes: int = 100):
    both_success = max(route_count - rescues - harms, 0)
    predictions = torch.tensor([1] * route_count + [0] * (episodes - route_count), dtype=torch.int64)
    outcomes = torch.tensor(
        [0] * rescues
        + [1] * harms
        + [2] * both_success
        + [3] * max(episodes - route_count, 0),
        dtype=torch.int64,
    )[:episodes]
    return direct_policy_metrics(
        predictions=predictions,
        outcomes=outcomes,
        stop_cost=100.0,
        branch_cost=200.0,
    )


def test_root_classifier_enforces_strict_economic_and_enrichment_gates():
    good = {
        "episodes": 100,
        "route_count": 10,
        "policy_solutions": 55,
        "policy_flops": 11000.0,
        "stop_successes": 45,
        "branch_successes": 55,
        "stop_flops": 10000.0,
        "branch_flops": 20000.0,
        "selected_rescues": 8,
        "selected_harms": 1,
        "selected_both_success": 1,
        "selected_both_failure": 0,
        "raw_rescues": 10,
        "provenance_closed": True,
    }
    assert classify_representation_root(good) == REPRESENTATION_ECONOMICALLY_IDENTIFIABLE

    partial = dict(good, selected_rescues=1, selected_harms=2)
    assert classify_representation_root(partial) == REPRESENTATION_PARTIAL

    failed = dict(good, policy_solutions=44)
    assert classify_representation_root(failed) == REPRESENTATION_NOT_IDENTIFIABLE


def test_budget_and_cross_classifiers_recompute_and_prioritize_frontier():
    root = {
        "episodes": 100,
        "route_count": 10,
        "policy_solutions": 55,
        "policy_flops": 11000.0,
        "stop_successes": 45,
        "branch_successes": 55,
        "stop_flops": 10000.0,
        "branch_flops": 20000.0,
        "selected_rescues": 8,
        "selected_harms": 1,
        "selected_both_success": 1,
        "selected_both_failure": 0,
        "raw_rescues": 10,
        "provenance_closed": True,
    }
    budget_class, pooled = classify_representation_budget([root, root, root, root])
    assert budget_class == REPRESENTATION_RECURRENTLY_IDENTIFIABLE
    assert pooled["policy_solutions"] == 220

    cross = classify_cross_budget(
        {
            "60": {V9_MEAN_BASELINE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE,
                   FULL_PREBRANCH_STATE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE,
                   EXECUTION_FRONTIER_STATE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE},
            "120": {V9_MEAN_BASELINE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE,
                    FULL_PREBRANCH_STATE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE,
                    EXECUTION_FRONTIER_STATE: REPRESENTATION_RECURRENTLY_IDENTIFIABLE},
        }
    )
    assert cross["representation_cross_classifications"][EXECUTION_FRONTIER_STATE] == CROSS_BUDGET_IDENTIFIABLE
    assert cross["decision"] == "FRONTIER_SIGNAL_IDENTIFIED"
    assert cross["authorized_representation_view"] == EXECUTION_FRONTIER_STATE
    assert cross["successor_design_authorized"] is True
    assert cross["mechanism_successor_authorized"] is False

    failed_cross = classify_cross_budget(
        {
            "60": {name: "REPRESENTATION_NOT_IDENTIFIABLE" for name in REPRESENTATION_DIMENSIONS},
            "120": {name: "REPRESENTATION_NOT_IDENTIFIABLE" for name in REPRESENTATION_DIMENSIONS},
        }
    )
    assert all(
        value == CROSS_BUDGET_NOT_IDENTIFIABLE
        for value in failed_cross["representation_cross_classifications"].values()
    )
    assert failed_cross["decision"] == "REPRESENTATION_SIGNAL_NOT_ESTABLISHED"
    assert failed_cross["successor_design_authorized"] is False
