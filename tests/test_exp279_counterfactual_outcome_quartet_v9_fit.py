from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    PRIMARY_FAMILY,
    STUDENT_OPTIMIZER,
    deterministic_student_seed,
    fit_quartet_student,
    fit_stop_utility,
)


def test_v9_fit_contract_is_deterministic_and_fixed() -> None:
    roots = ["fit-b", "fit-a"]
    seed_a = deterministic_student_seed(PRIMARY_FAMILY, 60, 2, roots)
    seed_b = deterministic_student_seed(PRIMARY_FAMILY, 60, 2, list(reversed(roots)))
    assert seed_a == seed_b

    g = torch.Generator().manual_seed(7)
    x = torch.randn(512, 144, generator=g)
    y = torch.tensor([0, 1, 2, 3] * 128, dtype=torch.long)
    first = fit_quartet_student(x, y, train_replicates=60, canonical_index=2, fit_roots=roots)
    second = fit_quartet_student(x, y, train_replicates=60, canonical_index=2, fit_roots=roots)

    assert first["optimizer"] == STUDENT_OPTIMIZER
    assert first["steps"] == 200
    assert first["batch_size"] == 512
    assert first["class_counts"] == {
        "RESCUE": 128,
        "HARM": 128,
        "BOTH_SUCCESS": 128,
        "BOTH_FAILURE": 128,
    }
    assert first["initial_digest"] == second["initial_digest"]
    assert first["final_digest"] == second["final_digest"]


def test_v9_fit_stop_utility_includes_student_cost() -> None:
    stop = torch.tensor([True, False, True, False], dtype=torch.bool)
    result = fit_stop_utility(stop, stop_accounted_flops=100, student_accounted_flops=10)
    assert result["episodes"] == 4
    assert result["solutions"] == 2
    assert result["total_accounted_flops"] == 440
    assert result["utility"] == pytest.approx(2 / 440)
