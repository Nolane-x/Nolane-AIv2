from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _analysis_api():
    try:
        from nolane_ai.experiments.exp286_confirmatory_analysis import (
            ALPHA,
            MESI_RELATIVE_REDUCTION,
            PRIMARY_ENDPOINT,
            SOLUTION_FLOOR_DIFFERENCE,
            build_exp286_confirmatory_analysis,
            decide_exp286_confirmatory_outcome,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 frozen confirmatory analysis court is missing")
    return (
        ALPHA,
        MESI_RELATIVE_REDUCTION,
        PRIMARY_ENDPOINT,
        SOLUTION_FLOOR_DIFFERENCE,
        build_exp286_confirmatory_analysis,
        decide_exp286_confirmatory_outcome,
    )


def test_exp286_analysis_surface_freezes_protocol_thresholds() -> None:
    (
        alpha,
        mesi,
        primary_endpoint,
        solution_floor,
        build_analysis,
        decide,
    ) = _analysis_api()

    assert alpha == 0.05
    assert mesi == 0.15
    assert primary_endpoint == "accounted_reasoning_flops_to_verified_solution"
    assert solution_floor == -0.005
    assert callable(build_analysis)
    assert callable(decide)


def test_exp286_decision_rule_promotes_only_when_primary_and_protected_bounds_pass() -> None:
    *_, decide = _analysis_api()

    assert (
        decide(
            primary_lower=0.15,
            primary_upper=0.20,
            solution_lower=-0.005,
            solution_upper=0.01,
        )
        == "PROMOTE_TO_NEXT_STAGE"
    )
    assert (
        decide(
            primary_lower=0.149,
            primary_upper=0.20,
            solution_lower=-0.005,
            solution_upper=0.01,
        )
        == "HOLD_UNSTABLE"
    )
    assert (
        decide(
            primary_lower=0.15,
            primary_upper=0.20,
            solution_lower=-0.006,
            solution_upper=0.01,
        )
        == "HOLD_UNSTABLE"
    )


def test_exp286_decision_rule_kills_when_upper_bound_cannot_reach_mesi_or_floor() -> None:
    *_, decide = _analysis_api()

    assert (
        decide(
            primary_lower=0.10,
            primary_upper=0.149,
            solution_lower=0.0,
            solution_upper=0.01,
        )
        == "KILL_SUBSYSTEM"
    )
    assert (
        decide(
            primary_lower=0.15,
            primary_upper=0.20,
            solution_lower=-0.02,
            solution_upper=-0.006,
        )
        == "KILL_SUBSYSTEM"
    )
