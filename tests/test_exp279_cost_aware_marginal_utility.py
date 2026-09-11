from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    ROUTING_SUPERVISION,
    _hybrid_cost_aware_marginal_utility_teacher,
)


def test_cost_aware_teacher_penalizes_equal_quality_branch_for_extra_compute() -> None:
    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    logits = torch.tensor(
        [
            [[2.0, -1.0], [-1.0, 2.0]],
            [[-1.0, 2.0], [2.0, -1.0]],
        ],
        dtype=torch.float32,
    )

    teacher = _hybrid_cost_aware_marginal_utility_teacher(
        logits,
        logits.clone(),
        targets,
        stop_accounted_flops_per_episode=100.0,
        branch_accounted_flops_per_episode=200.0,
    )

    assert teacher.shape == (2,)
    assert torch.all(teacher < 0.5)
    assert torch.allclose(teacher, torch.full_like(teacher, 1.0 / 3.0), atol=1e-6)


def test_cost_aware_teacher_routes_only_when_soft_exact_gain_beats_cost_ratio() -> None:
    targets = torch.tensor([[0, 0]], dtype=torch.long)
    stop_logits = torch.zeros(1, 2, 2, dtype=torch.float32)
    branch_logits = torch.tensor(
        [[[2.0, 0.0], [2.0, 0.0]]],
        dtype=torch.float32,
    )

    teacher = _hybrid_cost_aware_marginal_utility_teacher(
        stop_logits,
        branch_logits,
        targets,
        stop_accounted_flops_per_episode=100.0,
        branch_accounted_flops_per_episode=200.0,
    )

    assert float(teacher[0]) > 0.5


def test_cost_aware_teacher_is_half_at_exact_soft_utility_break_even() -> None:
    targets = torch.tensor([[0]], dtype=torch.long)
    stop_logits = torch.tensor([[[math.log(1.0 / 3.0), 0.0]]], dtype=torch.float32)
    branch_logits = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)

    teacher = _hybrid_cost_aware_marginal_utility_teacher(
        stop_logits,
        branch_logits,
        targets,
        stop_accounted_flops_per_episode=100.0,
        branch_accounted_flops_per_episode=200.0,
    )

    assert float(teacher[0]) == pytest.approx(0.5, abs=1e-6)
    assert teacher.requires_grad is False


def test_cost_aware_teacher_rejects_invalid_costs() -> None:
    targets = torch.tensor([[0]], dtype=torch.long)
    logits = torch.zeros(1, 1, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="accounted FLOPs"):
        _hybrid_cost_aware_marginal_utility_teacher(
            logits,
            logits,
            targets,
            stop_accounted_flops_per_episode=0.0,
            branch_accounted_flops_per_episode=10.0,
        )


def test_cost_aware_teacher_contract_keeps_frozen_route_decision_boundary() -> None:
    teacher = ROUTING_SUPERVISION["hybrid_route_teacher"]
    assert ROUTING_SUPERVISION["episode_targets"]["hybrid"] == "cost_aware_soft_marginal_utility"
    assert teacher["target"] == "soft_exact_utility_branch_share"
    assert teacher["soft_exact_surrogate"] == "exp_sum_log_target_probability"
    assert teacher["utility"] == "soft_exact_probability_divided_by_accounted_flops"
    assert teacher["temperature"] == 1.0
    assert teacher["cost_source"] == "sealed_pair_audit_hybrid_stop_and_branch_ledgers"
    assert teacher["uses_same_paired_training_targets"] is True
    assert teacher["evaluation_targets_used_for_routing"] is False
    assert teacher["decision_threshold_changed"] is False
