from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


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


def test_analytic_calibration_separates_prior_from_ranking_score() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import calibrated_route_probability

    prior = 0.025
    break_even = 0.5103974800459145

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
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import calibrated_route_probability

    prior = 0.025
    break_even = 0.5103974800459145
    posterior_odds_at_break_even = break_even / (1.0 - break_even)
    prior_odds = prior / (1.0 - prior)
    required_lr = posterior_odds_at_break_even / prior_odds

    score = calibrated_route_probability(
        log_likelihood_ratio=math.log(required_lr),
        natural_rescue_prior=prior,
        break_even_probability=break_even,
    )

    assert score == pytest.approx(0.5, abs=1e-12)


def test_calibration_rejects_invalid_priors_without_threshold_tuning() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_calibration import calibrated_route_probability

    for prior in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="natural_rescue_prior"):
            calibrated_route_probability(
                log_likelihood_ratio=0.0,
                natural_rescue_prior=prior,
                break_even_probability=0.5103974800459145,
            )

    for break_even in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="break_even_probability"):
            calibrated_route_probability(
                log_likelihood_ratio=0.0,
                natural_rescue_prior=0.025,
                break_even_probability=break_even,
            )
