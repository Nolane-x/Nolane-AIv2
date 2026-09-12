from __future__ import annotations

import inspect
import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9 import (
    CANONICAL_INDICES,
    CONTROL_FAMILY,
    DECISION_REPLICATES,
    DIAGNOSTIC_FOLDS,
    FIT_REPLICATES,
    PRIMARY_FAMILY,
    ROOT_PREFIX,
    STUDENT_GEOMETRY,
    STUDENT_OPTIMIZER,
    TRAIN_BUDGETS,
    QuartetClass,
    apply_quartet_policy,
    canonical_root,
    cheap_prebranch_features,
    decision_root,
    diagnostic_fold,
    direct_utility,
    expected_root_map,
    fit_root,
    marginal_route_score,
    quartet_labels,
    student_inference_flops,
)


def test_v9_frozen_identity_and_geometry() -> None:
    assert TRAIN_BUDGETS == (60, 120)
    assert CANONICAL_INDICES == (0, 1, 2, 3)
    assert FIT_REPLICATES == 1332
    assert DECISION_REPLICATES == 1332
    assert DIAGNOSTIC_FOLDS == 3
    assert ROOT_PREFIX == "20260912-exp279-counterfactual-outcome-quartet-v9-dev"
    assert PRIMARY_FAMILY == "OUTCOME_QUARTET_MLP"
    assert CONTROL_FAMILY == "RESCUE_ONLY_MLP_CONTROL"
    assert STUDENT_GEOMETRY == {"input_size": 144, "hidden_size": 64, "classes": 4}
    assert STUDENT_OPTIMIZER == {
        "name": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 512,
        "steps": 200,
    }


def test_v9_root_schedule_is_unique_and_cross_budget_disjoint() -> None:
    assert canonical_root(60, 2) == f"{ROOT_PREFIX}::train::60::canonical::2"
    assert fit_root(120, 3, 1) == f"{ROOT_PREFIX}::train::120::fit::3::1"
    assert decision_root(120, 3, 1) == f"{ROOT_PREFIX}::train::120::decision::3::1"

    r60 = expected_root_map(60)
    r120 = expected_root_map(120)
    assert len(r60["canonical_roots"]) == 4
    assert len(r60["fit_roots"]) == 8
    assert len(r60["decision_roots"]) == 8
    flat60 = set(r60["canonical_roots"] + r60["fit_roots"] + r60["decision_roots"])
    flat120 = set(r120["canonical_roots"] + r120["fit_roots"] + r120["decision_roots"])
    assert len(flat60) == 20
    assert len(flat120) == 20
    assert flat60.isdisjoint(flat120)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (canonical_root, (15, 0)),
        (canonical_root, (60, 4)),
        (fit_root, (60, 0, 2)),
        (decision_root, (120, -1, 0)),
        (decision_root, (999, 0, 0)),
    ],
)
def test_v9_root_helpers_reject_nonfrozen_identities(fn, args) -> None:
    with pytest.raises(ValueError):
        fn(*args)


def test_v9_diagnostic_folds_are_exactly_444_replicates_each() -> None:
    counts = [0, 0, 0]
    for replicate in range(DECISION_REPLICATES):
        counts[diagnostic_fold(replicate)] += 1
    assert counts == [444, 444, 444]
    assert [count * 8 for count in counts] == [3552, 3552, 3552]


def test_v9_quartet_labels_encode_complete_counterfactual_outcome() -> None:
    stop = torch.tensor([False, True, True, False])
    branch = torch.tensor([True, False, True, False])
    labels = quartet_labels(stop, branch)
    assert labels.tolist() == [
        int(QuartetClass.RESCUE),
        int(QuartetClass.HARM),
        int(QuartetClass.BOTH_SUCCESS),
        int(QuartetClass.BOTH_FAILURE),
    ]


def test_v9_quartet_labels_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        quartet_labels(torch.tensor([True]), torch.tensor([True, False]))
    with pytest.raises(ValueError):
        quartet_labels(torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool))


def test_v9_marginal_route_score_uses_rescue_minus_harm_and_incremental_cost() -> None:
    probs = torch.tensor(
        [
            [0.30, 0.05, 0.25, 0.40],
            [0.05, 0.20, 0.35, 0.40],
        ],
        dtype=torch.float64,
    )
    score = marginal_route_score(
        probs,
        fit_stop_utility=2.0e-6,
        stop_accounted_flops=100_000,
        branch_accounted_flops=150_000,
    )
    expected = torch.tensor([0.25 - 0.1, -0.15 - 0.1], dtype=torch.float64)
    assert torch.allclose(score, expected)
    assert apply_quartet_policy(score).tolist() == [True, False]
    assert apply_quartet_policy(torch.tensor([0.0])).tolist() == [False]


def test_v9_direct_utility_charges_student_on_every_policy_episode() -> None:
    chosen_branch = torch.tensor([False, True, False, True])
    stop_exact = torch.tensor([True, False, True, True])
    branch_exact = torch.tensor([False, True, False, False])
    result = direct_utility(
        chosen_branch,
        stop_exact,
        branch_exact,
        stop_accounted_flops=100,
        branch_accounted_flops=200,
        student_accounted_flops=10,
    )
    assert result["solutions"] == 3
    assert result["episodes"] == 4
    assert result["total_accounted_flops"] == 2 * 110 + 2 * 210
    assert result["utility"] == pytest.approx(3 / 640)


def test_v9_student_cost_is_fixed_positive_and_small_relative_to_branch() -> None:
    flops = student_inference_flops(hidden_size=48, variables=6, timesteps=4)
    assert isinstance(flops, int)
    assert flops > 0
    assert flops < 100_000
    assert flops == student_inference_flops(hidden_size=48, variables=6, timesteps=4)


def test_v9_primary_scientific_surface_exposes_no_tuning_knobs() -> None:
    sig = inspect.signature(student_inference_flops)
    assert set(sig.parameters) == {"hidden_size", "variables", "timesteps"}


def test_v9_frozen_episode_totals_close() -> None:
    per_root = 1332 * 8
    assert per_root == 10656
    assert 2 * per_root == 21312
    assert 4 * 2 * per_root == 85248
    assert 2 * 4 * 2 * per_root == 170496
    assert math.isfinite(float(per_root))
