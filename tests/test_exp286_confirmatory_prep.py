from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))


def _frozen_experiment() -> dict:
    protocol = _protocol()
    return next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-286")


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


def test_exp286_gate_a_plans_from_real_paired_development_without_consuming_challenge() -> None:
    pytest.importorskip("torch")

    from nolane_ai.experiments.exp286_confirmatory_prep import (
        build_exp286_confirmatory_prep,
        validate_exp286_confirmatory_prep,
    )
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    execution = run_exp286_paired_development(
        root_seed="exp286-confirmatory-prep-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=32,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=5,
        decoys=2,
        max_search_steps=8,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    registry = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=execution["protocol_digest"],
        model_audit=_audit(),
        exp286_pair_audit=execution["resource_match"]["pair_audit"],
        exp286_execution_artifact=execution,
    )

    prep = build_exp286_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest="a" * 64,
        familywise_alpha=_protocol()["global_sample_size_plan"]["familywise_alpha"],
    )

    assert prep["schema"] == "NLM-EXP-286-CONFIRMATORY-PREP-V1"
    assert prep["experiment_id"] == "EXP-286"
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["confirmatory_data_consumed"] is False
    assert prep["challenge_materialized"] is False
    assert prep["decision_rule_executed"] is False

    frozen = prep["frozen_analysis"]
    assert frozen["primary_endpoint"] == "accounted_reasoning_flops_to_verified_solution"
    assert frozen["primary_direction"] == "lower"
    assert frozen["effect_type"] == "paired_log_cost_ratio_relative_reduction"
    assert frozen["mesi_relative_reduction"] == pytest.approx(0.15)
    assert frozen["familywise_alpha"] == pytest.approx(0.05)
    assert "bootstrap_samples" not in frozen
    assert frozen["multiplicity_family"] == "CONFLICT_VALUE"
    assert frozen["protected_solution_floor"] == "oracle_conflict_core >= chronological_failure - 0.005"

    pilot = prep["pilot_summary"]
    assert pilot["source_lane"] == "DEVELOPMENT_PILOT_ONLY"
    assert pilot["n"] == 32
    assert pilot["paired_log_cost_sd"] >= 0.0

    freeze = prep["sample_size_freeze"]
    assert freeze["power_target"] == pytest.approx(0.90)
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
    assert validate_exp286_confirmatory_prep(prep) == []
