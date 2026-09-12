from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    CONTROL_FAMILY,
    STUDENT_OPTIMIZER,
    combine_decision_root_metrics,
    fit_rescue_control,
    heldout_policy_metrics,
)


def _metric(route: torch.Tensor) -> dict[str, object]:
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


def test_v9_combines_exactly_two_decision_roots_from_sufficient_statistics() -> None:
    a = _metric(torch.tensor([False, False, False, False, True, True, False, False]))
    b = _metric(torch.tensor([False, False, False, False, True, True, False, False]))
    combined = combine_decision_root_metrics([a, b])
    assert combined["episodes"] == 16
    assert combined["routed_episodes"] == 4
    assert combined["selected_rescues"] == 4
    assert combined["selected_harms"] == 0
    assert combined["policy_solutions"] == a["policy_solutions"] + b["policy_solutions"]
    assert combined["policy_utility"] == pytest.approx(
        combined["policy_solutions"] / combined["policy_total_accounted_flops"]
    )
    with pytest.raises(ValueError):
        combine_decision_root_metrics([a])


def test_v9_rescue_control_fit_is_deterministic_and_binary() -> None:
    g = torch.Generator().manual_seed(11)
    x = torch.randn(512, 144, generator=g)
    quartet = torch.tensor([0, 1, 2, 3] * 128, dtype=torch.long)
    first = fit_rescue_control(x, quartet, train_replicates=120, canonical_index=1, fit_roots=["b", "a"])
    second = fit_rescue_control(x, quartet, train_replicates=120, canonical_index=1, fit_roots=["a", "b"])
    assert first["family"] == CONTROL_FAMILY
    assert first["positive_count"] == 128
    assert first["negative_count"] == 384
    assert first["optimizer"] == STUDENT_OPTIMIZER
    assert first["initial_digest"] == second["initial_digest"]
    assert first["final_digest"] == second["final_digest"]
