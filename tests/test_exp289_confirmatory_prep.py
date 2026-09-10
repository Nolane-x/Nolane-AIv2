from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))


def _frozen_experiment() -> dict:
    return next(item for item in _protocol()["experiments"] if item["experiment_id"] == "EXP-289")


def _development_execution_and_registry(eval_replicates: int = 32):
    pytest.importorskip("torch")

    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from tests.test_exp289_registry_integration import _model_audit, _protocol as _registry_protocol

    execution = run_exp289_paired_development(
        root_seed="exp289-confirmatory-prep-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=eval_replicates,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        restarts=3,
        variables=6,
        decoys=2,
        max_search_steps=12,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    registry = build_neural_arm_registry(
        protocol=_registry_protocol(),
        protocol_digest=execution["protocol_digest"],
        model_audit=_model_audit(),
        exp289_pair_audit=execution["resource_match"]["pair_audit"],
        exp289_execution_artifact=execution,
    )
    return execution, registry


def test_exp289_confirmatory_prep_freezes_stage_a_authority_without_opening_challenge() -> None:
    from nolane_ai.experiments.exp289_confirmatory_prep import frozen_exp289_gate_a_contract

    contract = frozen_exp289_gate_a_contract()

    assert contract["experiment_id"] == "EXP-289"
    assert contract["primary_endpoint"] == "repeat_dead_end_rate"
    assert contract["primary_direction"] == "lower"
    assert contract["mesi_relative_reduction"] == pytest.approx(0.25)
    assert contract["power_target"] == pytest.approx(0.90)
    assert contract["familywise_alpha"] == pytest.approx(0.05)
    assert contract["min_n"] == 32
    assert contract["max_n"] == 128
    assert contract["paired"] is True
    assert contract["multiplicity_family"] == "LOCAL_NOGOOD"
    assert contract["zero_opportunity_policy"] == "retain_raw_episode_exclude_from_rder_denominator"
    assert contract["epsilon_denominator_rescue"] is False
    assert contract["protected_overprune_ceiling"] == pytest.approx(0.005)
    assert contract["protected_solution_floor"] == "local_nogood >= no_nogood - 0.01"
    assert contract["pilot_reuse_as_confirmatory"] is False
    assert contract["challenge_seed_materialized"] is False
    assert contract["confirmatory_data_consumed"] is False


def test_exp289_gate_a_plans_from_real_paired_development_without_consuming_challenge() -> None:
    from nolane_ai.experiments.exp289_confirmatory_prep import (
        build_exp289_confirmatory_prep,
        validate_exp289_confirmatory_prep,
    )

    execution, registry = _development_execution_and_registry(32)
    prep = build_exp289_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest="a" * 64,
        familywise_alpha=_protocol()["global_sample_size_plan"]["familywise_alpha"],
    )

    assert prep["schema"] == "NLM-EXP-289-CONFIRMATORY-PREP-V1"
    assert prep["experiment_id"] == "EXP-289"
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["confirmatory_data_consumed"] is False
    assert prep["challenge_materialized"] is False
    assert prep["decision_rule_executed"] is False

    frozen = prep["frozen_analysis"]
    assert frozen["primary_endpoint"] == "repeat_dead_end_rate"
    assert frozen["primary_direction"] == "lower"
    assert frozen["effect_type"] == "paired_relative_rder_reduction"
    assert frozen["mesi_relative_reduction"] == pytest.approx(0.25)
    assert frozen["familywise_alpha"] == pytest.approx(0.05)
    assert frozen["multiplicity_family"] == "LOCAL_NOGOOD"
    assert frozen["zero_opportunity_policy"] == "retain_raw_episode_exclude_from_rder_denominator"
    assert frozen["epsilon_denominator_rescue"] is False
    assert frozen["protected_overprune_ceiling"] == pytest.approx(0.005)
    assert frozen["protected_solution_floor"] == "local_nogood >= no_nogood - 0.01"

    pilot = prep["pilot_summary"]
    assert pilot["source_lane"] == "DEVELOPMENT_PILOT_ONLY"
    assert pilot["n"] == 32
    assert pilot["eligible_paired_rder_replicates"] == 32
    assert pilot["paired_relative_rder_reduction_sd"] >= 0.0

    freeze = prep["sample_size_freeze"]
    assert freeze["power_target"] == pytest.approx(0.90)
    assert freeze["familywise_alpha"] == pytest.approx(0.05)
    assert freeze["min_n"] == 32
    assert freeze["max_n"] == 128
    assert freeze["paired"] is True
    assert freeze["pilot_reuse_as_confirmatory"] is False
    assert freeze["unclamped_required_n"] >= 1
    if freeze["unclamped_required_n"] > 128:
        assert prep["status"] == "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
        assert prep["confirmatory_ready"] is False
        assert freeze["confirmatory_n"] is None
        assert prep["confirmatory_lineage"]["reserved_replicate_ids"] == []
    else:
        assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"
        assert prep["confirmatory_ready"] is True
        assert freeze["confirmatory_n"] == max(32, freeze["unclamped_required_n"])
        assert len(prep["confirmatory_lineage"]["reserved_replicate_ids"]) == freeze["confirmatory_n"]

    lineage = prep["confirmatory_lineage"]
    assert lineage["lane"] == "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED"
    assert lineage["pilot_reuse_forbidden"] is True
    assert lineage["seed_materialization_status"] == "NOT_EXECUTED"
    assert set(lineage["reserved_replicate_ids"]).isdisjoint(execution["evaluation"]["per_replicate"][index]["replicate"] for index in range(32))
    assert validate_exp289_confirmatory_prep(prep) == []


def test_exp289_paired_effect_refuses_zero_baseline_denominator_without_epsilon_rescue() -> None:
    from nolane_ai.experiments.exp289_confirmatory_prep import _paired_relative_rder_effects

    rows = [
        {
            "no_nogood": {"repeat_dead_end_rate": 0.0},
            "local_nogood": {"repeat_dead_end_rate": 0.0},
        }
    ]
    with pytest.raises(ValueError, match="baseline RDER denominator"):
        _paired_relative_rder_effects(rows)


def test_exp289_gate_a_rejects_insufficient_pilot_replicates() -> None:
    from nolane_ai.experiments.exp289_confirmatory_prep import build_exp289_confirmatory_prep

    execution, registry = _development_execution_and_registry(3)
    with pytest.raises(ValueError, match="at least 32 paired pilot replicates"):
        build_exp289_confirmatory_prep(
            experiment=_frozen_experiment(),
            execution_artifact=execution,
            arm_registry=registry,
            analysis_code_digest="a" * 64,
            familywise_alpha=0.05,
        )
