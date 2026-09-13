from __future__ import annotations

import pytest

from nolane_ai.experiments.exp301_compute import (
    COMPUTE_LEDGER_VERSION,
    EXP301_MAX_RELATIVE_FLOP_MISMATCH,
    account_arm_flops,
    match_common_compute,
    validate_compute_match,
)


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EFFORTS = (1, 2, 4, 8, 12, 16)


def test_compute_ledger_version_and_tolerance_are_frozen() -> None:
    assert COMPUTE_LEDGER_VERSION == "exp301-flops-v1"
    assert EXP301_MAX_RELATIVE_FLOP_MISMATCH == pytest.approx(0.002)


def test_accounted_compute_strictly_increases_with_effort_for_every_arm() -> None:
    for arm in ARMS:
        totals = [
            account_arm_flops(arm, effort_multiplier=effort, sequence_length=64).total_flops
            for effort in EFFORTS
        ]
        assert totals == sorted(totals)
        assert len(set(totals)) == len(totals)


def test_recurrent_loops_are_never_free() -> None:
    one = account_arm_flops("C_NRS_CORE", effort_multiplier=1, sequence_length=64)
    two = account_arm_flops("C_NRS_CORE", effort_multiplier=2, sequence_length=64)

    assert two.repeated_core_flops == 2 * one.repeated_core_flops
    assert two.conditioning_flops == 2 * one.conditioning_flops
    assert two.total_flops > one.total_flops


def test_fixed_restarts_pay_for_each_restart_but_only_one_output_projection() -> None:
    one = account_arm_flops("A_FIXED", effort_multiplier=1, sequence_length=64)
    four = account_arm_flops("A_FIXED", effort_multiplier=4, sequence_length=64)

    assert four.repeated_core_flops == 4 * one.repeated_core_flops
    assert four.restart_code_flops == 4 * one.restart_code_flops
    assert four.output_projection_flops == one.output_projection_flops
    assert four.aggregation_flops > 0
    assert four.total_flops < 4 * one.total_flops


def test_common_compute_matches_all_three_arms_within_frozen_tolerance() -> None:
    for sequence_length in (8, 32, 64, 128):
        for effort in EFFORTS:
            match = match_common_compute(
                effort_multiplier=effort,
                sequence_length=sequence_length,
            )
            assert match.status == "VALID_COMPUTE_MATCH"
            assert match.relative_mismatch <= EXP301_MAX_RELATIVE_FLOP_MISMATCH
            assert tuple(match.flops_by_arm) == ARMS


def test_match_validator_fails_closed_when_any_arm_exceeds_tolerance() -> None:
    status, mismatch = validate_compute_match(
        {
            "A_FIXED": 100_000,
            "B_LOOP_SIMPLE": 100_050,
            "C_NRS_CORE": 101_000,
        }
    )

    assert mismatch > EXP301_MAX_RELATIVE_FLOP_MISMATCH
    assert status == "INVALID_COMPUTE_MATCH"


def test_compute_ledger_rejects_unknown_arm_effort_or_invalid_sequence_length() -> None:
    with pytest.raises(ValueError, match="arm"):
        account_arm_flops("UNKNOWN", effort_multiplier=1, sequence_length=64)
    with pytest.raises(ValueError, match="effort"):
        account_arm_flops("A_FIXED", effort_multiplier=3, sequence_length=64)
    with pytest.raises(ValueError, match="sequence_length"):
        account_arm_flops("A_FIXED", effort_multiplier=1, sequence_length=0)
