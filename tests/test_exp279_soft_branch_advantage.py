from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    ROUTING_SUPERVISION,
    _hybrid_branch_advantage_teacher,
)


def test_soft_branch_advantage_teacher_tracks_per_episode_training_loss_advantage() -> None:
    targets = torch.tensor([[0, 1], [0, 1], [0, 1]])

    # Episode 0: forced branch is clearly better than stop.
    # Episode 1: forced branch is clearly worse than stop.
    # Episode 2: exact tie.
    stop_logits = torch.tensor(
        [
            [[0.1, 1.2], [1.3, 0.0]],
            [[2.0, -1.0], [-1.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    branch_logits = torch.tensor(
        [
            [[2.0, -1.0], [-1.0, 2.0]],
            [[0.1, 1.2], [1.3, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    teacher = _hybrid_branch_advantage_teacher(stop_logits, branch_logits, targets)

    assert teacher.shape == (3,)
    assert float(teacher[0]) > 0.5
    assert float(teacher[1]) < 0.5
    assert float(teacher[2]) == pytest.approx(0.5, abs=1e-7)
    assert torch.all((teacher > 0.0) & (teacher < 1.0))


def test_soft_branch_advantage_teacher_contract_preserves_frozen_route_threshold() -> None:
    teacher = ROUTING_SUPERVISION["hybrid_route_teacher"]
    assert teacher["target"] == "sigmoid_per_episode_stop_ce_minus_forced_branch_ce"
    assert teacher["temperature"] == 1.0
    assert teacher["uses_same_paired_training_targets"] is True
    assert teacher["evaluation_targets_used_for_routing"] is False
    assert teacher["decision_threshold_changed"] is False
