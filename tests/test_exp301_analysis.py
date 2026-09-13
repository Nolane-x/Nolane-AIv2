from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp301_analysis import (
    DECISION_INVALID,
    DECISION_KILL,
    DECISION_PROMOTE,
    REASONING_FAMILIES,
    normalized_rcg,
    paired_hierarchical_bootstrap_rcg,
    reduce_exp301,
)
from nolane_ai.experiments.exp301_compute import account_arm_flops, match_common_compute
from nolane_ai.experiments.exp301_evaluation import Exp301EvaluationRow


ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
EFFORTS = (1, 2, 4, 8, 12, 16)
ROOTS = (0, 1, 2, 3)


def _row(
    *,
    arm: str,
    family: str,
    root: int,
    effort: int,
    index: int,
    success: bool,
    invalid: bool = False,
    compute_match_status: str | None = None,
) -> Exp301EvaluationRow:
    sequence_length = 32
    flops = account_arm_flops(
        arm,
        effort_multiplier=effort,
        sequence_length=sequence_length,
    ).total_flops
    status = compute_match_status or match_common_compute(
        effort_multiplier=effort,
        sequence_length=sequence_length,
    ).status
    return Exp301EvaluationRow(
        arm_id=arm,
        family=family,
        root=root,
        effort_multiplier=effort,
        content_id=f"r{root}-{family}-e{effort}-i{index}",
        prediction_digest=f"{index:064x}",
        verified_success=success,
        invalid_output=invalid,
        resident_parameters=10_000_000,
        accounted_flops=flops,
        compute_match_status=status,
        unseen_depth_diagnostic=effort in (12, 16),
        primary_family_gain_eligible=effort in (1, 2, 4, 8),
    )


def _success(index: int, rate_tenths: int) -> bool:
    return index < rate_tenths


def _full_rows(
    *,
    c_reasoning_by_root: tuple[int, int, int, int] = (8, 8, 8, 8),
    a_reasoning: int = 2,
    b_reasoning: int = 4,
    a_language: int = 9,
    c_language: int = 9,
    b_language: int = 9,
) -> list[Exp301EvaluationRow]:
    rows: list[Exp301EvaluationRow] = []
    for root in ROOTS:
        for family in FAMILIES:
            for effort in EFFORTS:
                for index in range(10):
                    if family == "language-sequence-control":
                        rates = {
                            "A_FIXED": a_language,
                            "B_LOOP_SIMPLE": b_language,
                            "C_NRS_CORE": c_language,
                        }
                    else:
                        rates = {
                            "A_FIXED": a_reasoning,
                            "B_LOOP_SIMPLE": b_reasoning,
                            "C_NRS_CORE": c_reasoning_by_root[root],
                        }
                    for arm in ARMS:
                        rows.append(
                            _row(
                                arm=arm,
                                family=family,
                                root=root,
                                effort=effort,
                                index=index,
                                success=_success(index, rates[arm]),
                            )
                        )
    return rows


def test_normalized_rcg_is_constant_success_gap_for_flat_curves() -> None:
    rows = []
    for effort in EFFORTS:
        for index in range(10):
            rows.append(
                _row(
                    arm="A_FIXED",
                    family="algorithmic-sequence-transform",
                    root=0,
                    effort=effort,
                    index=index,
                    success=_success(index, 5),
                )
            )
            rows.append(
                _row(
                    arm="C_NRS_CORE",
                    family="algorithmic-sequence-transform",
                    root=0,
                    effort=effort,
                    index=index,
                    success=_success(index, 6),
                )
            )

    assert normalized_rcg(rows, candidate_arm="C_NRS_CORE", rival_arm="A_FIXED") == pytest.approx(0.1)


def test_bootstrap_is_deterministic_for_fixed_seed_and_positive_fixture() -> None:
    rows = _full_rows()
    first = paired_hierarchical_bootstrap_rcg(rows, samples=200, seed=301)
    second = paired_hierarchical_bootstrap_rcg(rows, samples=200, seed=301)

    assert first == second
    assert first[0] > 0.0
    assert first[1] > first[0]


def test_reducer_promotes_only_when_full_conjunction_passes() -> None:
    result = reduce_exp301(
        _full_rows(),
        bootstrap_samples=200,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )

    assert result.decision == DECISION_PROMOTE
    assert result.aggregate_rcg >= 0.05
    assert result.rcg_ci_low > 0.0
    assert result.qualifying_reasoning_family_count >= 2
    assert result.positive_root_count >= 3
    assert result.protected_floors["all_clear"] is True
    assert result.qualifying_effort in (1, 2, 4, 8)
    assert set(REASONING_FAMILIES) == {
        "iterative-grid-and-maze",
        "algorithmic-sequence-transform",
        "generator-heldout-abstract-transformation",
    }


def test_language_floor_failure_blocks_promotion_even_with_large_reasoning_gain() -> None:
    result = reduce_exp301(
        _full_rows(a_language=10, c_language=7, b_language=9),
        bootstrap_samples=100,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )

    assert result.protected_floors["language_control"] is False
    assert result.decision == DECISION_KILL


def test_only_two_positive_roots_cannot_promote() -> None:
    result = reduce_exp301(
        _full_rows(c_reasoning_by_root=(8, 8, 1, 1)),
        bootstrap_samples=200,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )

    assert result.positive_root_count == 2
    assert result.decision == DECISION_KILL


def test_compute_invalidity_yields_invalid_court_not_kill() -> None:
    rows = _full_rows()
    rows[0] = replace(rows[0], compute_match_status="INVALID_COMPUTE_MATCH")

    result = reduce_exp301(
        rows,
        bootstrap_samples=50,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )

    assert result.decision == DECISION_INVALID
    assert result.protected_floors["compute_match"] is False


def test_missing_paired_arm_row_is_invalid_court() -> None:
    rows = _full_rows()
    removed = rows.pop()
    assert removed.arm_id == "C_NRS_CORE"

    result = reduce_exp301(
        rows,
        bootstrap_samples=50,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )

    assert result.decision == DECISION_INVALID
    assert "pair" in result.reason.lower() or "complete" in result.reason.lower()


def test_external_leakage_or_scaffold_floor_blocks_scientific_promotion() -> None:
    leakage = reduce_exp301(
        _full_rows(),
        bootstrap_samples=50,
        bootstrap_seed=301,
        challenge_leakage_clear=False,
        hidden_scaffold_clear=True,
        representation_stability_ok=True,
    )
    scaffold = reduce_exp301(
        _full_rows(),
        bootstrap_samples=50,
        bootstrap_seed=301,
        challenge_leakage_clear=True,
        hidden_scaffold_clear=False,
        representation_stability_ok=True,
    )

    assert leakage.decision == DECISION_INVALID
    assert scaffold.decision == DECISION_INVALID
