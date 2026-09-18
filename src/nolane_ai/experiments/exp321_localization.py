from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .exp321_contract import DISPOSITIONS, THRESHOLDS


@dataclass(frozen=True, slots=True)
class EffortSummary:
    effort: int
    teacher_forced_token_accuracy: float
    teacher_forced_full_answer_exact: float
    greedy_exact: float
    masked_greedy_exact: float
    masked_wrong_target_recovery: float


def _unit(value: float, *, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")


def validate_effort_summary(summary: EffortSummary) -> None:
    if summary.effort not in (1, 2, 4, 8):
        raise ValueError("effort must be one of (1,2,4,8)")
    for label in (
        "teacher_forced_token_accuracy",
        "teacher_forced_full_answer_exact",
        "greedy_exact",
        "masked_greedy_exact",
        "masked_wrong_target_recovery",
    ):
        _unit(getattr(summary, label), label=label)


def reduce_localization(
    summaries: Mapping[int, EffortSummary],
    *,
    reproduction_valid: bool,
) -> str:
    if not reproduction_valid:
        return "INVALID_LOCALIZATION"
    if set(summaries) != {1, 2, 4, 8}:
        return "INVALID_LOCALIZATION"
    try:
        for effort, summary in summaries.items():
            if effort != summary.effort:
                return "INVALID_LOCALIZATION"
            validate_effort_summary(summary)
    except (TypeError, ValueError):
        return "INVALID_LOCALIZATION"

    baseline = summaries[4]
    effort_mismatch = any(
        summary.greedy_exact - baseline.greedy_exact >= THRESHOLDS.effort_effect_min
        or (
            summary.teacher_forced_full_answer_exact
            - baseline.teacher_forced_full_answer_exact
            >= THRESHOLDS.effort_effect_min
        )
        for effort, summary in summaries.items()
        if effort != 4
    )
    unused_vocab = (
        baseline.masked_greedy_exact - baseline.greedy_exact
        >= THRESHOLDS.masked_exact_gain_min
        or baseline.masked_wrong_target_recovery
        >= THRESHOLDS.masked_wrong_target_recovery_min
    )
    teacher_forced_insufficient = (
        baseline.teacher_forced_token_accuracy
        < THRESHOLDS.teacher_forced_token_accuracy
        or baseline.teacher_forced_full_answer_exact
        < THRESHOLDS.teacher_forced_full_answer_exact
    )
    rollout_exposure = (
        not teacher_forced_insufficient
        and baseline.greedy_exact < THRESHOLDS.greedy_exact
        and not unused_vocab
    )

    causes = sum((effort_mismatch, unused_vocab, rollout_exposure))
    if teacher_forced_insufficient and not effort_mismatch and not unused_vocab:
        decision = "TEACHER_FORCED_FOUNDATION_INSUFFICIENT"
    elif causes >= 2 or (
        teacher_forced_insufficient and (effort_mismatch or unused_vocab)
    ):
        decision = "MIXED_TRAINING_STACK_FAILURE"
    elif effort_mismatch:
        decision = "EFFORT_MISMATCH_EVIDENT"
    elif unused_vocab:
        decision = "UNUSED_VOCAB_COMPETITION_EVIDENT"
    elif rollout_exposure:
        decision = "ROLLOUT_EXPOSURE_BOTTLENECK"
    else:
        decision = "NO_DOMINANT_LOCALIZED_CAUSE"

    if decision not in DISPOSITIONS:
        raise RuntimeError("EXP-321 reducer emitted an unknown disposition")
    return decision
