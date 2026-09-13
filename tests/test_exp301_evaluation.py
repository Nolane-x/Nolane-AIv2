from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp301_evaluation import (
    PRIMARY_TRAINED_EFFORTS,
    UNSEEN_DEPTH_EFFORTS,
    aggregate_verified_success,
    commit_prediction,
    evaluate_protected_floors,
    score_committed_prediction,
    validate_prediction_commitment,
)
from nolane_ai.experiments.exp301_worlds import generate_world_instance


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")


def _world(family: str = "algorithmic-sequence-transform", *, index: int = 1):
    return generate_world_instance(family=family, root=0, split="development", index=index)


def test_prediction_is_committed_before_verifier_scoring() -> None:
    world = _world()
    commitment = commit_prediction(
        world,
        arm_id="C_NRS_CORE",
        root=0,
        effort_multiplier=4,
        candidate_answer=world.canonical_answer,
    )

    assert len(commitment.prediction_digest) == 64
    validate_prediction_commitment(commitment)
    row = score_committed_prediction(commitment, world)
    assert row.verified_success is True
    assert row.invalid_output is False
    assert row.prediction_digest == commitment.prediction_digest


def test_tampered_prediction_commitment_is_rejected() -> None:
    world = _world()
    commitment = commit_prediction(
        world,
        arm_id="A_FIXED",
        root=0,
        effort_multiplier=2,
        candidate_answer=world.canonical_answer,
    )
    tampered = replace(commitment, candidate_answer="tampered")

    with pytest.raises(ValueError, match="digest"):
        validate_prediction_commitment(tampered)
    with pytest.raises(ValueError, match="digest"):
        score_committed_prediction(tampered, world)


def test_wrong_or_empty_outputs_remain_in_ledger_as_failures() -> None:
    world = _world(index=2)
    wrong = commit_prediction(
        world,
        arm_id="B_LOOP_SIMPLE",
        root=0,
        effort_multiplier=1,
        candidate_answer="wrong",
    )
    empty = commit_prediction(
        world,
        arm_id="C_NRS_CORE",
        root=0,
        effort_multiplier=1,
        candidate_answer="",
    )

    wrong_row = score_committed_prediction(wrong, world)
    empty_row = score_committed_prediction(empty, world)
    assert wrong_row.verified_success is False
    assert wrong_row.invalid_output is False
    assert empty_row.verified_success is False
    assert empty_row.invalid_output is True


def test_scoring_rejects_instance_identity_mismatch() -> None:
    first = _world(index=1)
    second = _world(index=2)
    commitment = commit_prediction(
        first,
        arm_id="C_NRS_CORE",
        root=0,
        effort_multiplier=4,
        candidate_answer=first.canonical_answer,
    )
    with pytest.raises(ValueError, match="content_id"):
        score_committed_prediction(commitment, second)


def test_unseen_depths_are_diagnostic_and_cannot_alone_count_as_primary_family_gain() -> None:
    assert PRIMARY_TRAINED_EFFORTS == (1, 2, 4, 8)
    assert UNSEEN_DEPTH_EFFORTS == (12, 16)
    for effort in UNSEEN_DEPTH_EFFORTS:
        world = _world(index=effort)
        row = score_committed_prediction(
            commit_prediction(
                world,
                arm_id="C_NRS_CORE",
                root=0,
                effort_multiplier=effort,
                candidate_answer=world.canonical_answer,
            ),
            world,
        )
        assert row.unseen_depth_diagnostic is True
        assert row.primary_family_gain_eligible is False


def test_aggregate_verified_success_keeps_family_root_effort_and_arm_separate() -> None:
    rows = []
    for arm in ARMS:
        for index in range(2):
            world = _world(index=index)
            answer = world.canonical_answer if index == 0 or arm == "C_NRS_CORE" else "wrong"
            rows.append(
                score_committed_prediction(
                    commit_prediction(
                        world,
                        arm_id=arm,
                        root=0,
                        effort_multiplier=4,
                        candidate_answer=answer,
                    ),
                    world,
                )
            )
    aggregate = aggregate_verified_success(rows)
    assert aggregate[("A_FIXED", "algorithmic-sequence-transform", 0, 4)].verified_success == pytest.approx(0.5)
    assert aggregate[("B_LOOP_SIMPLE", "algorithmic-sequence-transform", 0, 4)].verified_success == pytest.approx(0.5)
    assert aggregate[("C_NRS_CORE", "algorithmic-sequence-transform", 0, 4)].verified_success == pytest.approx(1.0)


def test_protected_floors_detect_language_and_invalid_output_regressions() -> None:
    rows = []
    family = "language-sequence-control"
    for index in range(200):
        world = _world(family, index=index)
        a_answer = world.canonical_answer
        c_answer = world.canonical_answer if index < 190 else ""  # 5% invalid/failure
        for arm, answer in (("A_FIXED", a_answer), ("C_NRS_CORE", c_answer)):
            rows.append(
                score_committed_prediction(
                    commit_prediction(
                        world,
                        arm_id=arm,
                        root=0,
                        effort_multiplier=4,
                        candidate_answer=answer,
                    ),
                    world,
                )
            )

    floors = evaluate_protected_floors(rows)
    assert floors["resident_parameter_identity"] is True
    assert floors["compute_match"] is True
    assert floors["language_control"] is False
    assert floors["invalid_output"] is False
    assert floors["all_clear"] is False
