from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from nolane_ai.experiments.exp279_rescue_anchor_replay import (
    ROUTING_SUPERVISION,
    _append_rescue_anchor_states,
    _rescue_anchor_replay_loss,
)


def test_negative_only_batch_replays_prior_positive_rescue_pressure() -> None:
    probabilities = torch.tensor([0.10, 0.20, 0.30], requires_grad=True)
    rescue_targets = torch.zeros(3)
    replay_positive_probabilities = torch.tensor([0.25], requires_grad=True)

    loss = _rescue_anchor_replay_loss(
        probabilities,
        rescue_targets,
        replay_positive_probabilities,
    )

    expected_positive = F.binary_cross_entropy(
        replay_positive_probabilities,
        torch.ones_like(replay_positive_probabilities),
    )
    expected_negative = F.binary_cross_entropy(
        probabilities,
        torch.zeros_like(probabilities),
    )
    assert torch.allclose(loss, 0.5 * (expected_positive + expected_negative))


def test_negative_only_batch_before_first_anchor_has_zero_router_gradient() -> None:
    probabilities = torch.tensor([0.15, 0.25, 0.35], requires_grad=True)
    rescue_targets = torch.zeros(3)

    loss = _rescue_anchor_replay_loss(
        probabilities,
        rescue_targets,
        torch.empty(0),
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert probabilities.grad is not None
    assert torch.equal(probabilities.grad, torch.zeros_like(probabilities))


def test_rescue_anchor_storage_keeps_only_detached_positive_states() -> None:
    routing_states = torch.randn(4, 6, 8, requires_grad=True)
    rescue_targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    anchors: list[torch.Tensor] = []

    _append_rescue_anchor_states(anchors, routing_states, rescue_targets)

    assert len(anchors) == 1
    assert anchors[0].shape == (2, 6, 8)
    assert anchors[0].requires_grad is False
    assert torch.equal(anchors[0], routing_states.detach()[[1, 3]])


def test_rescue_anchor_contract_preserves_frozen_decision_threshold_and_provenance() -> None:
    teacher = ROUTING_SUPERVISION["hybrid_route_teacher"]
    replay = teacher["rescue_anchor_replay"]

    assert teacher["positive"] == "stop_exact_failure_and_forced_branch_exact_success"
    assert teacher["negative"] == "otherwise"
    assert replay == {
        "enabled": True,
        "source": "prior_augmentation_training_rescue_routing_states",
        "storage": "detached_all_prior_positive_states",
        "negative_source": "current_augmentation_training_non_rescues",
        "negative_only_before_first_anchor": "zero_routing_gradient",
        "class_balance": "equal_positive_negative_mass_when_both_present",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
    }
    assert teacher["decision_threshold_changed"] is False
    assert teacher["evaluation_targets_used_for_routing"] is False
