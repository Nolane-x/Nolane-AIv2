from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_rescue_likelihood_ratio import (
    ROUTING_SUPERVISION,
    _branch_outcome_masks,
    _economic_rescue_score_from_probability,
    _pairwise_rescue_ranking_loss,
    _solve_prevalence_intercept,
    run_exp279_rescue_likelihood_ratio_development,
    validate_exp279_rescue_likelihood_ratio_development,
)


def test_pairwise_ranking_is_shift_invariant_and_pushes_rescues_above_non_rescues() -> None:
    positive = torch.tensor([-1.0, -0.5], requires_grad=True)
    negative = torch.tensor([0.5, 1.0], requires_grad=True)

    loss = _pairwise_rescue_ranking_loss(positive, negative)
    shifted = _pairwise_rescue_ranking_loss(positive.detach() + 13.0, negative.detach() + 13.0)
    loss.backward()

    assert loss.item() == pytest.approx(shifted.item(), abs=1e-7)
    assert positive.grad is not None
    assert negative.grad is not None
    assert bool((positive.grad < 0.0).all().item())
    assert bool((negative.grad > 0.0).all().item())


def test_pairwise_ranking_has_zero_gradient_until_both_classes_exist() -> None:
    only_positive = torch.tensor([0.2, 0.4], requires_grad=True)
    no_negative = torch.empty(0)
    loss = _pairwise_rescue_ranking_loss(only_positive, no_negative)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert only_positive.grad is not None
    assert torch.equal(only_positive.grad, torch.zeros_like(only_positive))


def test_prevalence_intercept_matches_natural_training_prevalence_without_tuning() -> None:
    ranking_logits = torch.tensor([-2.0, -0.5, 0.3, 1.1, 2.4], dtype=torch.float64)
    natural_prevalence = 0.08

    intercept = _solve_prevalence_intercept(
        ranking_logits,
        natural_positive_fraction=natural_prevalence,
    )
    calibrated = torch.sigmoid(ranking_logits + intercept)

    assert math.isfinite(intercept)
    assert calibrated.mean().item() == pytest.approx(natural_prevalence, abs=1e-10)


def test_economic_score_half_is_exact_incremental_cost_break_even() -> None:
    stop_flops = 110_046.0
    branch_flops = 224_766.0
    incremental_cost_ratio = (branch_flops - stop_flops) / branch_flops
    rescue_break_even = incremental_cost_ratio / (1.0 + incremental_cost_ratio)

    score = _economic_rescue_score_from_probability(
        torch.tensor([rescue_break_even]),
        incremental_cost_ratio=incremental_cost_ratio,
    )

    assert score.item() == pytest.approx(0.5, abs=1e-7)


def test_branch_outcome_masks_partition_rescue_harm_and_neutral_cases() -> None:
    targets = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)
    stop_logits = torch.tensor(
        [
            [[0.0, 2.0]],
            [[2.0, 0.0]],
            [[2.0, 0.0]],
            [[0.0, 2.0]],
        ]
    )
    branch_logits = torch.tensor(
        [
            [[2.0, 0.0]],
            [[0.0, 2.0]],
            [[2.0, 0.0]],
            [[0.0, 2.0]],
        ]
    )

    masks = _branch_outcome_masks(stop_logits, branch_logits, targets)

    assert masks["rescue"].tolist() == [True, False, False, False]
    assert masks["harm"].tolist() == [False, True, False, False]
    assert masks["both_success"].tolist() == [False, False, True, False]
    assert masks["both_failure"].tolist() == [False, False, False, True]
    total = sum(mask.to(torch.int64) for mask in masks.values())
    assert torch.equal(total, torch.ones_like(total))


def test_likelihood_ratio_contract_separates_discrimination_from_calibration() -> None:
    teacher = ROUTING_SUPERVISION["hybrid_route_teacher"]
    ranking = teacher["rescue_likelihood_ratio"]

    assert teacher["positive"] == "stop_exact_failure_and_forced_branch_exact_success"
    assert teacher["negative"] == "otherwise"
    assert teacher["decision_threshold_changed"] is False
    assert teacher["evaluation_targets_used_for_routing"] is False
    assert ranking == {
        "enabled": True,
        "source": "augmentation_training_only",
        "ranking_loss": "pairwise_logistic_rescue_over_all_non_rescue",
        "ranking_score": "mean_routing_logit_on_propagation_state",
        "calibration": "training_moment_matching_to_natural_rescue_prevalence",
        "economics": "incremental_branch_flops_divided_by_branch_flops",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
        "tunable_calibration_hyperparameters": False,
    }


def test_tiny_runner_preserves_development_boundary_and_emits_calibration_receipt() -> None:
    payload = run_exp279_rescue_likelihood_ratio_development(
        root_seed="exp279-likelihood-ratio-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=6,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["route_config"]["threshold"] == 0.5
    assert payload["evaluation"]["start_replicate"] == 100
    receipt = payload["training"]["rescue_likelihood_ratio"]
    assert receipt["source"] == "augmentation_training_only"
    assert receipt["natural_total_count"] == 6
    assert receipt["natural_positive_count"] <= receipt["natural_total_count"]
    assert receipt["evaluation_examples_used"] is False
    assert receipt["evaluation_targets_used"] is False
    assert receipt["decision_threshold_changed"] is False
    assert receipt["tunable_calibration_hyperparameters"] is False
    assert receipt["calibration_method"] in {
        "training_moment_matching_to_natural_rescue_prevalence",
        "degenerate_zero_positive_all_stop",
        "degenerate_all_positive_all_route",
    }
    assert validate_exp279_rescue_likelihood_ratio_development(payload) == []
