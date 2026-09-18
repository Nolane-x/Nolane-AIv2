from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp323_contract import (
    PARENT_DECAY_MODEL_STATE_DIGEST,
    PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
    PARENT_DECAY_RNG_STATE_DIGEST,
)
from nolane_ai.experiments.exp323_runtime import (
    apply_post2048_learning_rate,
    validate_reconstruction_digests,
)


def _optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_reconstruction_digest_authority_is_exact() -> None:
    payload = validate_reconstruction_digests(
        model_digest=PARENT_DECAY_MODEL_STATE_DIGEST,
        optimizer_digest=PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        rng_digest=PARENT_DECAY_RNG_STATE_DIGEST,
        nonfinite_events=0,
    )
    assert payload["verified"] is True
    assert payload["completed_step"] == 2048

    with pytest.raises(ValueError, match="replay"):
        validate_reconstruction_digests(
            model_digest="0" * 64,
            optimizer_digest=PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
            rng_digest=PARENT_DECAY_RNG_STATE_DIGEST,
            nonfinite_events=0,
        )


def test_hold_arm_preserves_exact_reconstructed_optimizer_state() -> None:
    optimizer = _optimizer()
    before = deepcopy(optimizer.state_dict())
    apply_post2048_learning_rate(optimizer, arm="HOLD_5E5")
    after = optimizer.state_dict()
    assert before["param_groups"] == after["param_groups"]
    for key in before["state"]:
        for state_key, before_value in before["state"][key].items():
            after_value = after["state"][key][state_key]
            if torch.is_tensor(before_value):
                assert torch.equal(before_value, after_value)
            else:
                assert before_value == after_value


def test_second_decay_changes_only_param_group_learning_rate() -> None:
    optimizer = _optimizer()
    before = deepcopy(optimizer.state_dict())
    apply_post2048_learning_rate(optimizer, arm="DECAY_2P5E5")
    after = optimizer.state_dict()

    for key in before["state"]:
        for state_key, before_value in before["state"][key].items():
            after_value = after["state"][key][state_key]
            if torch.is_tensor(before_value):
                assert torch.equal(before_value, after_value)
            else:
                assert before_value == after_value

    for b, a in zip(before["param_groups"], after["param_groups"]):
        for key in b:
            if key == "lr":
                assert b[key] == pytest.approx(5e-5)
                assert a[key] == pytest.approx(2.5e-5)
            else:
                assert b[key] == a[key]


def test_intervention_rejects_reconstructed_optimizer_drift() -> None:
    optimizer = _optimizer()
    optimizer.param_groups[0]["weight_decay"] = 0.0
    with pytest.raises(ValueError, match="weight decay"):
        apply_post2048_learning_rate(optimizer, arm="HOLD_5E5")
