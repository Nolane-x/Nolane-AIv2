from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp322_runtime import (
    apply_intervention_learning_rate,
    validate_parent_receipt,
)


def _parent():
    return SimpleNamespace(
        stage="A_SANITY",
        arm_id="A_FIXED",
        root=0,
        learning_rate=1e-4,
        cumulative_step=1024,
        artifact_digest=(
            "d874545fa677569e30849128df337e27823d8aa4c5335adb9cb422968096a280"
        ),
        model_state_digest=(
            "18d738a3845a470f80cbfcb39662f630a73195fd383da7f71c9e54474115c7fb"
        ),
    )


def test_parent_receipt_authority_is_exact_and_fail_closed() -> None:
    validate_parent_receipt(_parent())
    forged = _parent()
    forged.cumulative_step = 512
    with pytest.raises(ValueError, match="cumulative_step"):
        validate_parent_receipt(forged)
    forged = _parent()
    forged.arm_id = "C_NRS_CORE"
    with pytest.raises(ValueError, match="arm_id"):
        validate_parent_receipt(forged)


def _optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_hold_arm_does_not_mutate_inherited_learning_rate() -> None:
    optimizer = _optimizer()
    before = deepcopy(optimizer.state_dict())
    apply_intervention_learning_rate(optimizer, arm="HOLD_1E4")
    after = optimizer.state_dict()
    assert all(group["lr"] == pytest.approx(1e-4) for group in after["param_groups"])
    assert before == after


def test_decay_arm_changes_only_param_group_learning_rate() -> None:
    optimizer = _optimizer()
    before = deepcopy(optimizer.state_dict())
    apply_intervention_learning_rate(optimizer, arm="DECAY_5E5")
    after = optimizer.state_dict()

    assert before["state"].keys() == after["state"].keys()
    for key in before["state"]:
        for state_key, before_value in before["state"][key].items():
            after_value = after["state"][key][state_key]
            if torch.is_tensor(before_value):
                assert torch.equal(before_value, after_value)
            else:
                assert before_value == after_value

    assert len(before["param_groups"]) == len(after["param_groups"])
    for b, a in zip(before["param_groups"], after["param_groups"]):
        for key in b:
            if key == "lr":
                assert b[key] == pytest.approx(1e-4)
                assert a[key] == pytest.approx(5e-5)
            else:
                assert b[key] == a[key]


def test_intervention_rejects_optimizer_geometry_drift() -> None:
    optimizer = _optimizer()
    optimizer.param_groups[0]["weight_decay"] = 0.0
    with pytest.raises(ValueError, match="weight decay"):
        apply_intervention_learning_rate(optimizer, arm="HOLD_1E4")
