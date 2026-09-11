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


def test_hybrid_route_supervision_targets_only_recoverable_stop_failures() -> None:
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


def test_hybrid_route_supervision_contract_is_branch_usefulness_not_failure_only() -> None:
    assert ROUTING_SUPERVISION["episode_targets"]["hybrid"] == "branch_rescue_required"
    assert ROUTING_SUPERVISION["hybrid_route_teacher"] == {
        "positive": "stop_exact_failure_and_forced_branch_exact_success",
        "negative": "otherwise",
        "evaluation_targets_used_for_routing": False,
        "class_balance": "equal_positive_negative_mass_when_both_present",
        "decision_threshold_changed": False,
    }
