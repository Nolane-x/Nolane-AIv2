from __future__ import annotations

import math
from typing import Iterable


def _finite_floats(values: Iterable[float], *, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probabilities must lie strictly within (0, 1)")
    return math.log(probability / (1.0 - probability))


def _pairwise_auc(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    pairs = 0
    for pos in positive:
        for neg in negative:
            pairs += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / pairs


def summarize_score_court(
    *,
    log_likelihood_ratios: Iterable[float],
    rescue_targets: Iterable[bool],
    natural_rescue_prior: float,
    break_even_probability: float,
) -> dict[str, float | int | bool | str]:
    """Summarize a post-hoc discrimination-vs-economics diagnostic.

    Evaluation labels are used only to label already-produced score distributions.
    This helper neither trains/calibrates a router nor changes a decision threshold.
    """
    scores = _finite_floats(log_likelihood_ratios, name="log_likelihood_ratios")
    labels = list(rescue_targets)
    if len(scores) != len(labels):
        raise ValueError("scores and rescue targets must have identical length")
    if not all(isinstance(label, bool) for label in labels):
        raise ValueError("rescue_targets must be boolean")

    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        raise ValueError("score court requires both rescue and nonrescue labels")

    prior_log_odds = _logit(natural_rescue_prior)
    break_even_log_odds = _logit(break_even_probability)
    required = break_even_log_odds - prior_log_odds

    return {
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "posthoc_diagnostic_only": True,
        "evaluation_targets_used_for_training": False,
        "evaluation_targets_used_for_calibration": False,
        "decision_threshold_changed": False,
        "promotion_claimed": False,
        "n": len(scores),
        "rescue_count": len(positive),
        "nonrescue_count": len(negative),
        "natural_rescue_prior": float(natural_rescue_prior),
        "break_even_probability": float(break_even_probability),
        "required_log_likelihood_ratio": required,
        "rescue_mean_log_likelihood_ratio": sum(positive) / len(positive),
        "nonrescue_mean_log_likelihood_ratio": sum(negative) / len(negative),
        "rescue_fraction_above_required": sum(score >= required for score in positive) / len(positive),
        "nonrescue_fraction_above_required": sum(score >= required for score in negative) / len(negative),
        "pairwise_auc": _pairwise_auc(positive, negative),
    }
