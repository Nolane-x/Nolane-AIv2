from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9_scientific_runner import (
    FROZEN_ARM_GEOMETRY,
    FROZEN_CANONICAL_OPTIMIZER,
    FROZEN_ROUTE_THRESHOLD,
    FROZEN_WORLD_GEOMETRY,
    exact_outcome_tensors,
    run_exp279_counterfactual_outcome_quartet_v9_shard,
)


def test_v9_scientific_runner_exposes_no_tuning_knobs() -> None:
    signature = inspect.signature(run_exp279_counterfactual_outcome_quartet_v9_shard)
    assert set(signature.parameters) == {
        "train_replicates",
        "canonical_index",
        "protocol_digest",
        "code_digest",
        "scientific_branch_head",
        "executed_commit",
    }
    assert FROZEN_WORLD_GEOMETRY == {
        "batch_size": 8,
        "timesteps": 4,
        "variables": 6,
        "constraints": 3,
        "d_model": 64,
        "noise_std": 0.05,
    }
    assert FROZEN_ARM_GEOMETRY == {"hidden_size": 48, "target_parameters": 500_000}
    assert FROZEN_CANONICAL_OPTIMIZER == {"lr": 0.002, "weight_decay": 0.0}
    assert FROZEN_ROUTE_THRESHOLD == 0.5


def test_v9_exact_outcome_tensors_match_the_four_counterfactual_classes() -> None:
    targets = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])
    stop_logits = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    branch_logits = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    stop_exact, branch_exact = exact_outcome_tensors(stop_logits, branch_logits, targets)
    assert stop_exact.tolist() == [False, True, True, False]
    assert branch_exact.tolist() == [True, False, True, False]
