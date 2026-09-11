from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def pairwise_rescue_harm_loss(
    rescue_scores: torch.Tensor,
    harm_scores: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic ranking loss that orders rescue above harm."""
    if rescue_scores.shape != harm_scores.shape:
        raise ValueError("rescue_scores and harm_scores must have identical shape")
    if rescue_scores.numel() == 0:
        raise ValueError("rescue_scores and harm_scores must be non-empty")
    return F.softplus(harm_scores - rescue_scores).mean()


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(log_odds: float) -> float:
    if log_odds >= 0.0:
        tail = math.exp(-log_odds)
        return 1.0 / (1.0 + tail)
    head = math.exp(log_odds)
    return head / (1.0 + head)


def calibrated_route_probability(
    *,
    log_likelihood_ratio: float,
    natural_rescue_prior: float,
    break_even_probability: float,
) -> float:
    """Map discrimination evidence to a fixed-0.5 economic routing score.

    ``log_likelihood_ratio`` carries discrimination only. The natural rescue
    prior converts that evidence to posterior rescue odds. Subtracting the
    analytically supplied break-even log-odds makes a returned probability of
    exactly 0.5 correspond to posterior rescue probability exactly equal to the
    branch-cost break-even probability. No decision-threshold tuning occurs.
    """
    if not 0.0 < natural_rescue_prior < 1.0:
        raise ValueError("natural_rescue_prior must be strictly between 0 and 1")
    if not 0.0 < break_even_probability < 1.0:
        raise ValueError("break_even_probability must be strictly between 0 and 1")
    if not math.isfinite(log_likelihood_ratio):
        raise ValueError("log_likelihood_ratio must be finite")

    posterior_log_odds = _logit(natural_rescue_prior) + float(log_likelihood_ratio)
    economic_log_odds = posterior_log_odds - _logit(break_even_probability)
    return _sigmoid(economic_log_odds)
