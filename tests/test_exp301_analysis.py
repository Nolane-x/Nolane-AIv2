from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp301_analysis import (
    BOOTSTRAP_SEED,
    Exp301AnalysisSummary,
    compute_primary_rcg,
    decision_from_summary,
    paired_hierarchical_bootstrap,
    reduce_exp301,
)
from nolane_ai.experiments.exp301_evaluation import Exp301EvaluationRow


REASONING_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
)
LANGUAGE_FAMILY = "language-sequence-control"
PRIMARY_EFFORTS = (1, 2, 4, 8)


def _row(
    *,
    arm: str,
    family: str,
    root: int,
    effort: int,
    index: int,
    success: bool,
    invalid: bool = False,
    compute_status: str = "VALID_COMPUTE_MATCH",
) -> Exp301EvaluationRow:
    # FLOPs are deliberately a strict geometric series so normalized AUC has
    # an analytically transparent log2 axis in the reducer tests.
    flops = 1_000_000 * effort
    return Exp301EvaluationRow(
        arm_id=arm,
        family=family,
        root=root,
        effort_multiplier=effort,
        content_id=f"r{root}:{family}:i{index}",
        prediction_digest=f"{arm}:{root}:{family}:{effort}:{index}",
        verified_success=success,
        invalid_output=invalid,
        resident_parameters=10_000_000,
        accounted_flops=flops,
        compute_match_status=compute_status,
        unseen_depth_diagnostic=effort in (12, 16),
        primary_family_gain_eligible=effort in PRIMARY_EFFORTS,
    )


def _promotion_rows() -> list[Exp301EvaluationRow]:
    rows: list[Exp301EvaluationRow] = []
    for root in range(4):
        for family in (*REASONING_FAMILIES, LANGUAGE_FAMILY):
            for effort in PRIMARY_EFFORTS:
                for index in range(20):
                    if family == LANGUAGE_FAMILY:
                        a_success = index < 18
                        c_success = index < 18
                    else:
                        a_success = index < 8
                        c_success = index < 12
                    rows.append(
                        _row(
                            arm="A_FIXED",
                            family=family,
                            root=root,
                            effort=effort,
                            index=index,
                            success=a_success,
                        )
                    )
                    rows.append(
                        _row(
                            arm="C_NRS_CORE",
                            family=family,
                            root=root,
                            effort=effort,
                            index=index,
                            success=c_success,
                        )
                    )
    return rows


def test_primary_rcg_is_normalized_auc_difference_on_common_log_flop_axis() -> None:
    rows = _promotion_rows()

    result = compute_primary_rcg(rows)

    # Three reasoning families gain +0.20 while language is tied, therefore
    # the family-balanced aggregate gain is +0.15 at every primary budget.
    assert result.aggregate_rcg == pytest.approx(0.15, abs=1e-12)
    assert result.valid_efforts == PRIMARY_EFFORTS
    assert result.axis == "log2(accounted_flops)"
    assert result.normalized is True


def test_unseen_depth_rows_cannot_change_primary_rcg() -> None:
    rows = _promotion_rows()
    baseline = compute_primary_rcg(rows)
    for root in range(4):
        for family in (*REASONING_FAMILIES, LANGUAGE_FAMILY):
            for effort in (12, 16):
                for index in range(20):
                    rows.append(
                        _row(
                            arm="A_FIXED",
                            family=family,
                            root=root,
                            effort=effort,
                            index=index,
                            success=False,
                        )
                    )
                    rows.append(
                        _row(
                            arm="C_NRS_CORE",
                            family=family,
                            root=root,
                            effort=effort,
                            index=index,
                            success=True,
                        )
                    )

    after = compute_primary_rcg(rows)
    assert after.aggregate_rcg == baseline.aggregate_rcg
    assert after.valid_efforts == baseline.valid_efforts


def test_paired_hierarchical_bootstrap_is_deterministic_and_excludes_zero_for_fixture() -> None:
    rows = _promotion_rows()

    left = paired_hierarchical_bootstrap(rows, samples=300, seed=BOOTSTRAP_SEED)
    right = paired_hierarchical_bootstrap(rows, samples=300, seed=BOOTSTRAP_SEED)

    assert left == right
    assert left.samples == 300
    assert left.seed == BOOTSTRAP_SEED
    assert left.ci_low > 0.0
    assert left.ci_high >= left.ci_low


def test_full_promotion_fixture_satisfies_exact_conjunction() -> None:
    summary = reduce_exp301(_promotion_rows(), bootstrap_samples=300)

    assert summary.decision == "PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY"
    assert summary.aggregate_rcg >= 0.05
    assert summary.bootstrap_ci_low > 0.0
    assert summary.positive_roots >= 3
    assert summary.reasoning_families_ge_5pp >= 2
    assert summary.protected_floors_all_clear is True
    assert summary.exp302_implementation_authorized is False
    assert summary.scale_authorized is False


def test_decision_is_conjunctive_not_score_compensatory() -> None:
    passing = Exp301AnalysisSummary(
        decision="UNDECIDED",
        aggregate_rcg=0.10,
        bootstrap_ci_low=0.04,
        bootstrap_ci_high=0.16,
        positive_roots=4,
        reasoning_families_ge_5pp=3,
        language_regression=0.0,
        protected_floors_all_clear=True,
        infrastructure_valid=True,
        valid_primary_efforts=PRIMARY_EFFORTS,
        bootstrap_seed=BOOTSTRAP_SEED,
        exp302_implementation_authorized=False,
        scale_authorized=False,
    )
    assert decision_from_summary(passing) == "PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY"

    language_fail = replace(passing, language_regression=0.021)
    assert decision_from_summary(language_fail) == "KILL_H_RD_01"

    family_fail = replace(passing, reasoning_families_ge_5pp=1)
    assert decision_from_summary(family_fail) == "KILL_H_RD_01"

    root_fail = replace(passing, positive_roots=2)
    assert decision_from_summary(root_fail) == "KILL_H_RD_01"

    ci_touches_zero = replace(passing, bootstrap_ci_low=0.0)
    assert decision_from_summary(ci_touches_zero) == "KILL_H_RD_01"

    invalid = replace(passing, infrastructure_valid=False)
    assert decision_from_summary(invalid) == "INVALID_COURT"


def test_compute_mismatch_makes_court_invalid_not_negative() -> None:
    rows = _promotion_rows()
    rows[0] = replace(rows[0], compute_match_status="INVALID_COMPUTE_MATCH")

    summary = reduce_exp301(rows, bootstrap_samples=100)

    assert summary.decision == "INVALID_COURT"
    assert summary.infrastructure_valid is False


def test_missing_root_makes_court_invalid() -> None:
    rows = [row for row in _promotion_rows() if row.root != 3]

    summary = reduce_exp301(rows, bootstrap_samples=100)

    assert summary.decision == "INVALID_COURT"
    assert summary.infrastructure_valid is False


def test_valid_negative_kills_hypothesis_without_authorizing_successor() -> None:
    rows = _promotion_rows()
    rows = [
        replace(row, verified_success=False)
        if row.arm_id == "C_NRS_CORE" and row.family in REASONING_FAMILIES
        else row
        for row in rows
    ]

    summary = reduce_exp301(rows, bootstrap_samples=100)

    assert summary.decision == "KILL_H_RD_01"
    assert summary.infrastructure_valid is True
    assert summary.exp302_implementation_authorized is False
    assert summary.scale_authorized is False
