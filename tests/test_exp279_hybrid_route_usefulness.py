from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    ROUTING_SUPERVISION,
    _hybrid_branch_rescue_target,
)


def _logits(predictions: list[list[int]]) -> torch.Tensor:
    rows: list[list[list[float]]] = []
    for episode in predictions:
        rows.append([
            [8.0, -8.0] if prediction == 0 else [-8.0, 8.0]
            for prediction in episode
        ])
    return torch.tensor(rows, dtype=torch.float32)


def test_legacy_branch_rescue_target_still_marks_only_recoverable_stop_failures() -> None:
    targets = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )
    stop_logits = _logits(
        [
            [1, 0],  # stop fails; branch rescues -> route
            [1, 1],  # stop fails; branch also fails -> do not pay branch cost
            [1, 1],  # stop already succeeds -> do not route
            [1, 0],  # stop already succeeds even though branch will fail -> do not route
        ]
    )
    branch_logits = _logits(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0],
        ]
    )

    route_target = _hybrid_branch_rescue_target(stop_logits, branch_logits, targets)

    assert route_target.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_hybrid_route_supervision_contract_is_cost_aware_marginal_utility() -> None:
    assert ROUTING_SUPERVISION["episode_targets"]["hybrid"] == "cost_aware_soft_marginal_utility"
    assert ROUTING_SUPERVISION["hybrid_route_teacher"] == {
        "target": "soft_exact_utility_branch_share",
        "soft_exact_surrogate": "exp_sum_log_target_probability",
        "utility": "soft_exact_probability_divided_by_accounted_flops",
        "temperature": 1.0,
        "cost_source": "sealed_pair_audit_hybrid_stop_and_branch_ledgers",
        "uses_same_paired_training_targets": True,
        "evaluation_targets_used_for_routing": False,
        "decision_threshold_changed": False,
    }
