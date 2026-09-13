from __future__ import annotations

import pytest

from nolane_ai.experiments import exp301_compute as compute


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
EFFORTS = (1, 2, 4, 8, 12, 16)


def test_compute_ledger_version_and_tolerance_are_frozen() -> None:
    # V2 is a real semantic revision: scientific receipts must include the full
    # autoregressive decode cost rather than only one prompt forward pass.
    assert compute.COMPUTE_LEDGER_VERSION == "exp301-flops-v2"
    assert compute.EXP301_MAX_RELATIVE_FLOP_MISMATCH == pytest.approx(0.002)


def test_accounted_compute_strictly_increases_with_effort_for_every_arm() -> None:
    for arm in ARMS:
        totals = [
            compute.account_arm_flops(arm, effort_multiplier=effort, sequence_length=64).total_flops
            for effort in EFFORTS
        ]
        assert totals == sorted(totals)
        assert len(set(totals)) == len(totals)


def test_recurrent_loops_are_never_free() -> None:
    one = compute.account_arm_flops("C_NRS_CORE", effort_multiplier=1, sequence_length=64)
    two = compute.account_arm_flops("C_NRS_CORE", effort_multiplier=2, sequence_length=64)

    assert two.repeated_core_flops == 2 * one.repeated_core_flops
    assert two.conditioning_flops == 2 * one.conditioning_flops
    assert two.total_flops > one.total_flops


def test_fixed_restarts_pay_for_each_restart_but_only_one_output_projection() -> None:
    one = compute.account_arm_flops("A_FIXED", effort_multiplier=1, sequence_length=64)
    four = compute.account_arm_flops("A_FIXED", effort_multiplier=4, sequence_length=64)

    assert four.repeated_core_flops == 4 * one.repeated_core_flops
    assert four.restart_code_flops == 4 * one.restart_code_flops
    assert four.output_projection_flops == one.output_projection_flops
    assert four.aggregation_flops > 0
    assert four.total_flops < 4 * one.total_flops


def test_common_compute_matches_all_three_arms_within_frozen_tolerance() -> None:
    for sequence_length in (8, 32, 64, 128):
        for effort in EFFORTS:
            match = compute.match_common_compute(
                effort_multiplier=effort,
                sequence_length=sequence_length,
            )
            assert match.status == "VALID_COMPUTE_MATCH"
            assert match.relative_mismatch <= compute.EXP301_MAX_RELATIVE_FLOP_MISMATCH
            assert tuple(match.flops_by_arm) == ARMS


def test_autoregressive_receipt_sums_every_decode_prefix() -> None:
    assert hasattr(compute, "account_autoregressive_flops")
    receipt = compute.account_autoregressive_flops(
        "C_NRS_CORE",
        effort_multiplier=4,
        prompt_token_count=17,
        generated_token_count=5,
    )
    expected = sum(
        compute.account_arm_flops(
            "C_NRS_CORE",
            effort_multiplier=4,
            sequence_length=17 + offset,
        ).total_flops
        for offset in range(5)
    )
    assert receipt.prompt_token_count == 17
    assert receipt.generated_token_count == 5
    assert receipt.decode_sequence_lengths == (17, 18, 19, 20, 21)
    assert receipt.total_flops == expected


def test_autoregressive_compute_match_accounts_for_generation_length() -> None:
    assert hasattr(compute, "match_autoregressive_compute")
    one = compute.match_autoregressive_compute(
        effort_multiplier=8,
        prompt_token_count=31,
        generated_token_count=1,
    )
    seven = compute.match_autoregressive_compute(
        effort_multiplier=8,
        prompt_token_count=31,
        generated_token_count=7,
    )
    assert one.status == seven.status == "VALID_COMPUTE_MATCH"
    assert seven.flops_by_arm["A_FIXED"] > one.flops_by_arm["A_FIXED"]
    assert seven.flops_by_arm["C_NRS_CORE"] > one.flops_by_arm["C_NRS_CORE"]
    assert seven.relative_mismatch <= compute.EXP301_MAX_RELATIVE_FLOP_MISMATCH


def test_autoregressive_ledger_rejects_zero_or_negative_generation_steps() -> None:
    assert hasattr(compute, "account_autoregressive_flops")
    with pytest.raises(ValueError, match="generated_token_count"):
        compute.account_autoregressive_flops(
            "A_FIXED",
            effort_multiplier=1,
            prompt_token_count=10,
            generated_token_count=0,
        )


def test_match_validator_fails_closed_when_any_arm_exceeds_tolerance() -> None:
    status, mismatch = compute.validate_compute_match(
        {
            "A_FIXED": 100_000,
            "B_LOOP_SIMPLE": 100_050,
            "C_NRS_CORE": 101_000,
        }
    )

    assert mismatch > compute.EXP301_MAX_RELATIVE_FLOP_MISMATCH
    assert status == "INVALID_COMPUTE_MATCH"


def test_compute_ledger_rejects_unknown_arm_effort_or_invalid_sequence_length() -> None:
    with pytest.raises(ValueError, match="arm"):
        compute.account_arm_flops("UNKNOWN", effort_multiplier=1, sequence_length=64)
    with pytest.raises(ValueError, match="effort"):
        compute.account_arm_flops("A_FIXED", effort_multiplier=3, sequence_length=64)
    with pytest.raises(ValueError, match="sequence_length"):
        compute.account_arm_flops("A_FIXED", effort_multiplier=1, sequence_length=0)
