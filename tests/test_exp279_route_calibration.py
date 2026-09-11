from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    ROUTING_SUPERVISION,
    _balanced_binary_cross_entropy,
)


def test_hybrid_rescue_calibration_balances_positive_and_negative_teacher_mass() -> None:
    probabilities = torch.tensor([0.2, 0.2, 0.2, 0.2])
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0])

    loss = _balanced_binary_cross_entropy(probabilities, targets)

    positive_loss = torch.nn.functional.binary_cross_entropy(
        probabilities[:1], targets[:1]
    )
    negative_loss = torch.nn.functional.binary_cross_entropy(
        probabilities[1:], targets[1:]
    )
    expected = 0.5 * (positive_loss + negative_loss)
    assert torch.allclose(loss, expected)


def test_hybrid_rescue_calibration_contract_records_class_balance() -> None:
    calibration = ROUTING_SUPERVISION["hybrid_route_teacher"]
    assert calibration["class_balance"] == "equal_positive_negative_mass_when_both_present"
    assert calibration["decision_threshold_changed"] is False
