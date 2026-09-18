from __future__ import annotations

import pytest

from nolane_ai.experiments.exp321_contract import (
    AUTHORIZATION_FLAGS,
    BYTE_ID_END_INCLUSIVE,
    BYTE_ID_START,
    EFFORT_GRID,
    EOS_ID,
    EXP319_FINAL_EVIDENCE_DIGEST,
    SELECTED_ARM,
    SELECTED_LEARNING_RATE,
    SELECTED_STEP,
    UNUSED_ID_END_INCLUSIVE,
    UNUSED_ID_START,
    VOCAB_SIZE,
    preregistration_digest,
    preregistration_payload,
)
from nolane_ai.experiments.exp321_localization import (
    EffortSummary,
    reduce_localization,
    validate_world_population,
)


def _summary(
    effort: int,
    *,
    token: float = 1.0,
    teacher_exact: float = 1.0,
    greedy: float = 1.0,
    masked: float = 1.0,
    recovery: float = 0.0,
) -> EffortSummary:
    return EffortSummary(
        effort=effort,
        teacher_forced_token_accuracy=token,
        teacher_forced_full_answer_exact=teacher_exact,
        greedy_exact=greedy,
        masked_greedy_exact=masked,
        masked_wrong_target_recovery=recovery,
    )


def _grid(baseline: EffortSummary) -> dict[int, EffortSummary]:
    def matched(effort: int) -> EffortSummary:
        return EffortSummary(
            effort=effort,
            teacher_forced_token_accuracy=baseline.teacher_forced_token_accuracy,
            teacher_forced_full_answer_exact=baseline.teacher_forced_full_answer_exact,
            greedy_exact=baseline.greedy_exact,
            masked_greedy_exact=baseline.masked_greedy_exact,
            masked_wrong_target_recovery=baseline.masked_wrong_target_recovery,
        )

    return {
        1: matched(1),
        2: matched(2),
        4: baseline,
        8: matched(8),
    }


def test_preregistration_is_bound_to_exp319_negative_control_evidence() -> None:
    payload = preregistration_payload()
    assert payload["scientific_boundary"]["exp319_disposition"] == (
        "TRAINING_STACK_NOT_LEARNABLE"
    )
    assert EXP319_FINAL_EVIDENCE_DIGEST == (
        "4be2850f6c1d633af4cb6687e379c0011aaf8281075bea79821d0f2e038ffb07"
    )
    assert SELECTED_ARM == "A_FIXED"
    assert SELECTED_LEARNING_RATE == 1e-4
    assert SELECTED_STEP == 1024
    assert preregistration_digest() == (
        "b99f74880848dca1332114cd0ca99292c53fba5ef991eedaace126c68835fc90"
    )


def test_tokenizer_geometry_is_frozen_without_head_resize() -> None:
    assert VOCAB_SIZE == 4608
    assert EOS_ID == 3
    assert (BYTE_ID_START, BYTE_ID_END_INCLUSIVE) == (4, 259)
    assert (UNUSED_ID_START, UNUSED_ID_END_INCLUSIVE) == (260, 4607)
    assert EFFORT_GRID == (1, 2, 4, 8)


def test_every_authorization_remains_false() -> None:
    assert AUTHORIZATION_FLAGS
    assert not any(AUTHORIZATION_FLAGS.values())


def test_teacher_forced_failure_has_precedence_when_isolated() -> None:
    decision = reduce_localization(
        _grid(
            _summary(
                4,
                token=0.77,
                teacher_exact=0.28,
                greedy=0.28,
                masked=0.28,
            )
        ),
        reproduction_valid=True,
    )
    assert decision == "TEACHER_FORCED_FOUNDATION_INSUFFICIENT"


def test_rollout_exposure_requires_teacher_forced_floor() -> None:
    decision = reduce_localization(
        _grid(
            _summary(
                4,
                token=0.995,
                teacher_exact=0.95,
                greedy=0.30,
                masked=0.31,
            )
        ),
        reproduction_valid=True,
    )
    assert decision == "ROLLOUT_EXPOSURE_BOTTLENECK"


def test_effort_mismatch_is_detected_against_effort_four() -> None:
    summaries = _grid(
        _summary(4, token=0.995, teacher_exact=0.95, greedy=0.30, masked=0.30)
    )
    summaries[8] = _summary(8, greedy=0.70)
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "EFFORT_MISMATCH_EVIDENT"
    )


def test_unused_vocab_counterfactual_is_detected() -> None:
    decision = reduce_localization(
        _grid(
            _summary(
                4,
                token=0.995,
                teacher_exact=0.95,
                greedy=0.30,
                masked=0.60,
                recovery=0.30,
            )
        ),
        reproduction_valid=True,
    )
    assert decision == "UNUSED_VOCAB_COMPETITION_EVIDENT"


def test_invalid_reproduction_fail_closes() -> None:
    assert reduce_localization(
        _grid(_summary(4)),
        reproduction_valid=False,
    ) == "INVALID_LOCALIZATION"

def test_effort_plus_unused_vocab_is_mixed_not_rollout_double_count() -> None:
    summaries = _grid(
        _summary(
            4,
            token=0.995,
            teacher_exact=0.95,
            greedy=0.30,
            masked=0.60,
            recovery=0.30,
        )
    )
    summaries[8] = _summary(
        8,
        token=0.995,
        teacher_exact=0.95,
        greedy=0.70,
        masked=0.70,
        recovery=0.0,
    )
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "MIXED_TRAINING_STACK_FAILURE"
    )


def test_nonfinite_summary_fails_closed() -> None:
    summaries = _grid(_summary(4))
    summaries[4] = _summary(4, token=float("nan"))
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "INVALID_LOCALIZATION"
    )
    summaries = _grid(_summary(4))
    summaries[4] = _summary(4, greedy=float("inf"))
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "INVALID_LOCALIZATION"
    )


def test_missing_effort_fails_closed() -> None:
    summaries = _grid(_summary(4))
    summaries.pop(8)
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "INVALID_LOCALIZATION"
    )


def test_mismatched_effort_identity_fails_closed() -> None:
    summaries = _grid(_summary(4))
    summaries[8] = _summary(2)
    assert reduce_localization(summaries, reproduction_valid=True) == (
        "INVALID_LOCALIZATION"
    )


def test_teacher_forced_plus_unused_vocab_is_mixed() -> None:
    decision = reduce_localization(
        _grid(
            _summary(
                4,
                token=0.77,
                teacher_exact=0.28,
                greedy=0.28,
                masked=0.60,
                recovery=0.30,
            )
        ),
        reproduction_valid=True,
    )
    assert decision == "MIXED_TRAINING_STACK_FAILURE"

def _valid_population() -> tuple[tuple[str, str], ...]:
    families = (
        "algorithmic-sequence-transform",
        "generator-heldout-abstract-transformation",
        "iterative-grid-and-maze",
        "language-sequence-control",
    )
    return tuple(
        (family, f"{family}-{index}")
        for family in families
        for index in range(8)
    )


def test_frozen_world_population_accepts_exact_unique_geometry() -> None:
    validate_world_population(_valid_population())


def test_duplicate_world_population_fails_closed() -> None:
    identities = list(_valid_population())
    identities[-1] = identities[0]
    with pytest.raises(ValueError, match="unique|family population"):
        validate_world_population(tuple(identities))


def test_missing_world_population_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly 32"):
        validate_world_population(_valid_population()[:-1])


def test_family_population_drift_fails_closed() -> None:
    identities = list(_valid_population())
    identities[-1] = ("algorithmic-sequence-transform", "extra-unique")
    with pytest.raises(ValueError, match="family population"):
        validate_world_population(tuple(identities))

