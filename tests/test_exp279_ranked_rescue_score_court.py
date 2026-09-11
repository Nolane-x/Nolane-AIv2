from __future__ import annotations

import math

import pytest


def test_score_court_separates_discrimination_from_economic_calibration() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_score_court import summarize_score_court

    court = summarize_score_court(
        log_likelihood_ratios=[-1.0, 0.0, 1.0, 3.5],
        rescue_targets=[False, False, True, True],
        natural_rescue_prior=0.025,
        break_even_probability=0.3379226242024708,
    )

    expected_required = math.log(
        (0.3379226242024708 / (1.0 - 0.3379226242024708))
        / (0.025 / (1.0 - 0.025))
    )
    assert court["required_log_likelihood_ratio"] == pytest.approx(expected_required)
    assert court["rescue_mean_log_likelihood_ratio"] == pytest.approx(2.25)
    assert court["nonrescue_mean_log_likelihood_ratio"] == pytest.approx(-0.5)
    assert court["rescue_fraction_above_required"] == pytest.approx(0.5)
    assert court["nonrescue_fraction_above_required"] == pytest.approx(0.0)
    assert court["pairwise_auc"] == pytest.approx(1.0)


def test_score_court_is_posthoc_and_cannot_authorize_threshold_tuning() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_score_court import summarize_score_court

    court = summarize_score_court(
        log_likelihood_ratios=[-0.2, 0.1, 0.4],
        rescue_targets=[False, True, False],
        natural_rescue_prior=0.03,
        break_even_probability=0.3379226242024708,
    )

    assert court["evidence_level"] == "EV-E2"
    assert court["decision"] == "UNVERIFIED"
    assert court["posthoc_diagnostic_only"] is True
    assert court["evaluation_targets_used_for_training"] is False
    assert court["evaluation_targets_used_for_calibration"] is False
    assert court["decision_threshold_changed"] is False
    assert court["promotion_claimed"] is False


def test_score_court_fails_closed_on_degenerate_labels() -> None:
    from nolane_ai.experiments.exp279_ranked_rescue_score_court import summarize_score_court

    with pytest.raises(ValueError, match="both rescue and nonrescue"):
        summarize_score_court(
            log_likelihood_ratios=[0.0, 1.0],
            rescue_targets=[True, True],
            natural_rescue_prior=0.025,
            break_even_probability=0.3379226242024708,
        )
