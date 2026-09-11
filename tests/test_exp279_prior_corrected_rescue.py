from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_prior_corrected_rescue import (
    ROUTING_SUPERVISION,
    _case_control_sample_probability,
    _economic_rescue_score_from_probability,
    _natural_rescue_probability_from_score,
    _prior_corrected_replay_loss,
    run_exp279_prior_corrected_rescue_development,
    validate_exp279_prior_corrected_rescue_development,
)


def test_economic_score_half_is_exact_incremental_cost_break_even() -> None:
    stop_flops = 110_046.0
    branch_flops = 224_766.0
    incremental_cost_ratio = (branch_flops - stop_flops) / branch_flops
    rescue_break_even = incremental_cost_ratio / (1.0 + incremental_cost_ratio)

    score = _economic_rescue_score_from_probability(
        torch.tensor([rescue_break_even]),
        incremental_cost_ratio=incremental_cost_ratio,
    )
    recovered = _natural_rescue_probability_from_score(
        score,
        incremental_cost_ratio=incremental_cost_ratio,
    )

    assert score.item() == pytest.approx(0.5, abs=1e-7)
    assert recovered.item() == pytest.approx(rescue_break_even, abs=1e-7)


def test_case_control_transform_maps_natural_prior_to_sample_prior() -> None:
    natural_prior = 0.10
    sampled_prior = 0.50
    transformed = _case_control_sample_probability(
        torch.tensor([natural_prior]),
        natural_positive_fraction=natural_prior,
        sampled_positive_fraction=sampled_prior,
    )
    assert transformed.item() == pytest.approx(sampled_prior, abs=1e-7)


def test_balanced_replay_shared_score_gradient_is_zero_at_natural_prior_after_correction() -> None:
    stop_flops = 110_046.0
    branch_flops = 224_766.0
    incremental_cost_ratio = (branch_flops - stop_flops) / branch_flops
    natural_prior = 0.01
    score_value = _economic_rescue_score_from_probability(
        torch.tensor([natural_prior]),
        incremental_cost_ratio=incremental_cost_ratio,
    ).item()

    shared_score = torch.tensor(score_value, requires_grad=True)
    current_scores = shared_score.unsqueeze(0)
    current_targets = torch.tensor([0.0])
    replay_positive_scores = shared_score.unsqueeze(0)

    loss = _prior_corrected_replay_loss(
        current_scores,
        current_targets,
        replay_positive_scores,
        natural_positive_count=1,
        natural_total_count=100,
        stop_accounted_flops_per_episode=stop_flops,
        branch_accounted_flops_per_episode=branch_flops,
    )
    loss.backward()

    assert loss.item() == pytest.approx(-torch.log(torch.tensor(0.5)).item(), abs=1e-6)
    assert shared_score.grad is not None
    assert shared_score.grad.item() == pytest.approx(0.0, abs=1e-5)


def test_before_first_rescue_router_gradient_is_zero_but_not_rebalanced() -> None:
    scores = torch.tensor([0.20, 0.30, 0.40], requires_grad=True)
    targets = torch.zeros(3)

    loss = _prior_corrected_replay_loss(
        scores,
        targets,
        torch.empty(0),
        natural_positive_count=0,
        natural_total_count=3,
        stop_accounted_flops_per_episode=110_046.0,
        branch_accounted_flops_per_episode=224_766.0,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_prior_corrected_contract_keeps_frozen_threshold_and_training_only_provenance() -> None:
    teacher = ROUTING_SUPERVISION["hybrid_route_teacher"]
    calibration = teacher["prior_corrected_rescue_replay"]

    assert teacher["positive"] == "stop_exact_failure_and_forced_branch_exact_success"
    assert teacher["negative"] == "otherwise"
    assert teacher["decision_threshold_changed"] is False
    assert teacher["evaluation_targets_used_for_routing"] is False
    assert calibration == {
        "enabled": True,
        "source": "augmentation_training_only",
        "positive_replay": "detached_all_prior_rescue_routing_states",
        "natural_prior": "causal_cumulative_raw_augmentation_rescue_frequency",
        "sampling_correction": "exact_case_control_odds_correction",
        "economics": "incremental_branch_flops_divided_by_branch_flops",
        "negative_only_before_first_rescue": "zero_routing_gradient_but_count_in_natural_prior",
        "evaluation_examples_used": False,
        "evaluation_targets_used": False,
        "external_examples_added": False,
        "tunable_calibration_hyperparameters": False,
    }


def test_prior_corrected_runner_emits_fail_closed_development_receipt() -> None:
    artifact = run_exp279_prior_corrected_rescue_development(
        root_seed="exp279-prior-corrected-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=3,
        eval_start_replicate=700,
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

    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["route_config"]["threshold"] == 0.5
    assert artifact["training"]["routing_supervision"] == ROUTING_SUPERVISION

    replay = artifact["training"]["prior_corrected_rescue_replay"]
    assert replay["source"] == "augmentation_training_only"
    assert replay["natural_total_count"] == 6
    assert replay["natural_positive_count"] == replay["final_anchor_episodes"]
    assert replay["evaluation_examples_used"] is False
    assert replay["evaluation_targets_used"] is False
    assert replay["decision_threshold_changed"] is False
    assert replay["tunable_calibration_hyperparameters"] is False
    assert 0.0 < replay["incremental_cost_ratio"] < 1.0
    assert validate_exp279_prior_corrected_rescue_development(artifact) == []
