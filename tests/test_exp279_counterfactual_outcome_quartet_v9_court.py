from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    classify_quartet_root,
    heldout_policy_metrics,
)


def _surface(route: torch.Tensor) -> dict[str, object]:
    stop = torch.tensor([True, True, True, True, False, False, False, False])
    branch = torch.tensor([True, True, False, True, True, True, False, False])
    return heldout_policy_metrics(
        route,
        stop,
        branch,
        stop_accounted_flops=100,
        branch_accounted_flops=150,
        student_accounted_flops=1,
    )


def test_v9_root_pass_requires_real_direct_economic_gain() -> None:
    metrics = _surface(torch.tensor([False, False, False, False, True, True, False, False]))
    assert metrics["route_fraction"] == pytest.approx(0.25)
    assert metrics["selected_rescues"] == 2
    assert metrics["selected_harms"] == 0
    assert metrics["raw_rescue_prevalence"] == pytest.approx(0.25)
    assert metrics["selected_rescue_prevalence"] == pytest.approx(1.0)
    assert metrics["policy_utility"] > metrics["stop_baseline_utility"]
    assert metrics["policy_utility"] > metrics["branch_baseline_utility"]
    assert classify_quartet_root(metrics) == "QUARTET_ROOT_ECONOMIC"


def test_v9_root_rejects_stop_collapse_even_when_stop_is_strong() -> None:
    metrics = _surface(torch.zeros(8, dtype=torch.bool))
    assert classify_quartet_root(metrics) == "QUARTET_ROOT_NOT_ECONOMIC"


def test_v9_root_rejects_selected_harm_not_below_selected_rescue() -> None:
    metrics = _surface(torch.tensor([False, False, True, False, True, False, False, False]))
    assert metrics["selected_rescues"] == 1
    assert metrics["selected_harms"] == 1
    assert classify_quartet_root(metrics) == "QUARTET_ROOT_NOT_ECONOMIC"


def test_v9_root_rejects_open_evidence_boundary() -> None:
    metrics = _surface(torch.tensor([False, False, False, False, True, True, False, False]))
    metrics["evidence_boundary_closed"] = False
    assert classify_quartet_root(metrics) == "QUARTET_ROOT_NOT_ECONOMIC"
