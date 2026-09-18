from __future__ import annotations

import pytest

torch=pytest.importorskip("torch")

from nolane_ai.experiments.exp324_contract import FAMILIES,LEARNING_RATE
from nolane_ai.experiments.exp324_runtime import family_worlds,validate_optimizer_invariants


def test_family_world_filter_preserves_eight_unique_worlds() -> None:
    for family in FAMILIES:
        worlds=family_worlds(family)
        assert len(worlds)==8
        assert all(w.family==family for w in worlds)
        assert len({w.content_id for w in worlds})==8


def test_optimizer_invariants_require_exact_5e5_and_weight_decay() -> None:
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01)
    validate_optimizer_invariants(optimizer)
    optimizer.param_groups[0]["lr"]=2.5e-5
    with pytest.raises(ValueError,match="LR"): validate_optimizer_invariants(optimizer)
