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


def test_hard_rescue_target_remains_available_as_development_diagnostic() -> None:
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
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
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


def test_hybrid_route_supervision_contract_uses_soft_branch_advantage() -> None:
    assert ROUTING_SUPERVISION["episode_targets"]["hybrid"] == "soft_branch_advantage"
    assert ROUTING_SUPERVISION["hybrid_route_teacher"] == {
        "target": "sigmoid_per_episode_stop_ce_minus_forced_branch_ce",
        "temperature": 1.0,
        "uses_same_paired_training_targets": True,
        "evaluation_targets_used_for_routing": False,
        "decision_threshold_changed": False,
    }
