from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


STOP_FLOPS = 110_046.0
BRANCH_FLOPS = 224_766.0


def test_pairwise_ranking_loss_prefers_rescue_over_harm() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import pairwise_rescue_harm_loss

    rescue = torch.tensor([2.0, 1.5], requires_grad=True)
    harm = torch.tensor([-1.0, -0.5], requires_grad=True)
    good = pairwise_rescue_harm_loss(rescue, harm)
    bad = pairwise_rescue_harm_loss(harm, rescue)

    assert good.item() < bad.item()
    good.backward()
    assert torch.all(rescue.grad < 0)
    assert torch.all(harm.grad > 0)


def test_balanced_discriminator_logit_is_likelihood_ratio_scale() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
        balanced_log_likelihood_ratio_loss,
    )

    rescue = torch.tensor([0.0, 0.0], requires_grad=True)
    nonrescue = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
    loss = balanced_log_likelihood_ratio_loss(rescue, nonrescue)

    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-7)
    loss.backward()
    assert torch.all(rescue.grad < 0)
    assert torch.all(nonrescue.grad > 0)
    assert rescue.grad.abs().sum().item() == pytest.approx(
        nonrescue.grad.abs().sum().item(), abs=1e-7
    )


def test_ledger_break_even_is_rescue_probability_not_incremental_cost_ratio() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
        branch_rescue_break_even_probability,
    )

    incremental_cost_ratio = (BRANCH_FLOPS - STOP_FLOPS) / BRANCH_FLOPS
    break_even = branch_rescue_break_even_probability(
        stop_accounted_flops_per_episode=STOP_FLOPS,
        branch_accounted_flops_per_episode=BRANCH_FLOPS,
    )

    assert incremental_cost_ratio == pytest.approx(0.5103974800459145, abs=1e-15)
    assert break_even == pytest.approx(
        incremental_cost_ratio / (1.0 + incremental_cost_ratio), abs=1e-15
    )
    assert break_even == pytest.approx(0.3379226242024708, abs=1e-15)
    assert break_even != pytest.approx(incremental_cost_ratio)


def test_analytic_calibration_separates_prior_from_discrimination() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
        branch_rescue_break_even_probability,
        calibrated_route_probability,
    )

    prior = 0.025
    break_even = branch_rescue_break_even_probability(
        stop_accounted_flops_per_episode=STOP_FLOPS,
        branch_accounted_flops_per_episode=BRANCH_FLOPS,
    )

    neutral = calibrated_route_probability(
        log_likelihood_ratio=0.0,
        natural_rescue_prior=prior,
        break_even_probability=break_even,
    )
    strong = calibrated_route_probability(
        log_likelihood_ratio=math.log(100.0),
        natural_rescue_prior=prior,
        break_even_probability=break_even,
    )

    assert neutral < 0.5
    assert strong > neutral
    assert 0.0 < neutral < 1.0
    assert 0.0 < strong < 1.0


def test_fixed_half_threshold_is_exact_economic_break_even() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
        branch_rescue_break_even_probability,
        calibrated_route_probability,
    )

    prior = 0.025
    break_even = branch_rescue_break_even_probability(
        stop_accounted_flops_per_episode=STOP_FLOPS,
        branch_accounted_flops_per_episode=BRANCH_FLOPS,
    )
    posterior_odds_at_break_even = break_even / (1.0 - break_even)
    prior_odds = prior / (1.0 - prior)
    required_lr = posterior_odds_at_break_even / prior_odds

    score = calibrated_route_probability(
        log_likelihood_ratio=math.log(required_lr),
        natural_rescue_prior=prior,
        break_even_probability=break_even,
    )

    assert score == pytest.approx(0.5, abs=1e-12)


def test_calibration_rejects_invalid_priors_and_ledgers_without_threshold_tuning() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import (
        branch_rescue_break_even_probability,
        calibrated_route_probability,
    )

    break_even = branch_rescue_break_even_probability(
        stop_accounted_flops_per_episode=STOP_FLOPS,
        branch_accounted_flops_per_episode=BRANCH_FLOPS,
    )
    for prior in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="natural_rescue_prior"):
            calibrated_route_probability(
                log_likelihood_ratio=0.0,
                natural_rescue_prior=prior,
                break_even_probability=break_even,
            )

    for invalid in (
        {"stop_accounted_flops_per_episode": 0.0, "branch_accounted_flops_per_episode": 1.0},
        {"stop_accounted_flops_per_episode": 2.0, "branch_accounted_flops_per_episode": 1.0},
        {"stop_accounted_flops_per_episode": 1.0, "branch_accounted_flops_per_episode": 1.0},
    ):
        with pytest.raises(ValueError, match="branch_accounted_flops_per_episode"):
            branch_rescue_break_even_probability(**invalid)
