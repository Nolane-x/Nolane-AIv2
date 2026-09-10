from __future__ import annotations

import pytest


def test_exp286_confirmatory_prep_freezes_stage_a_authority_without_opening_challenge() -> None:
    from nolane_ai.experiments.exp286_confirmatory_prep import frozen_exp286_gate_a_contract

    contract = frozen_exp286_gate_a_contract()

    assert contract["experiment_id"] == "EXP-286"
    assert contract["primary_endpoint"] == "accounted_reasoning_flops_to_verified_solution"
    assert contract["primary_direction"] == "lower"
    assert contract["mesi_relative_reduction"] == pytest.approx(0.15)
    assert contract["power_target"] == pytest.approx(0.90)
    assert contract["min_n"] == 32
    assert contract["max_n"] == 128
    assert contract["paired"] is True
    assert contract["multiplicity_family"] == "CONFLICT_VALUE"
    assert contract["analysis_method"] == (
        "paired log-cost ratio and bootstrap CI; failures included as censored/scientific outcomes per frozen rule"
    )
    assert contract["protected_solution_floor"] == (
        "oracle_conflict_core >= chronological_failure - 0.005"
    )
    assert contract["pilot_reuse_as_confirmatory"] is False
    assert contract["challenge_seed_materialized"] is False
    assert contract["confirmatory_data_consumed"] is False
