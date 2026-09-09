from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
PRIMARY = "verified_utility_per_accounted_flop_on_structure_dense_stratum"


def _frozen_experiment() -> dict:
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    return next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-279")


def _stable_execution() -> dict:
    from nolane_ai.experiments.exp279_paired_runner import (
        _aggregate_rows,
        _artifact_digest,
        run_exp279_paired_development,
        validate_exp279_paired_development,
    )

    artifact = run_exp279_paired_development(
        root_seed="exp279-confirmatory-prep-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=33,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    rows = artifact["evaluation"]["per_replicate"]
    max_flops = max(
        float(row[arm]["accounted_flops_per_episode"])
        for row in rows
        for arm in ("propagation_only", "branch_only", "hybrid")
    )
    base = 0.10 / max_flops
    stratum_scale = {
        "PROPAGATION_FIT": 1.00,
        "BRANCH_FIT": 1.03,
        "MIXED_RESIDUAL": 0.97,
    }
    arm_scale = {
        "propagation_only": 1.00,
        "branch_only": 1.01,
        "hybrid": 1.12,
    }
    for row in rows:
        for arm in ("propagation_only", "branch_only", "hybrid"):
            utility = base * stratum_scale[row["stratum"]] * arm_scale[arm]
            flops = float(row[arm]["accounted_flops_per_episode"])
            row[arm]["verified_solution_rate"] = utility * flops
            row[arm][PRIMARY] = utility
    artifact["evaluation"]["aggregate"] = _aggregate_rows(rows)
    artifact["artifact_digest"] = _artifact_digest(artifact)
    assert validate_exp279_paired_development(artifact) == []
    return artifact


def _registry(execution: dict) -> dict:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
    from tests.test_neural_arm_registry import _audit, _protocol_subset

    return build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest=execution["protocol_digest"],
        model_audit=_audit(),
        exp279_pair_audit=execution["resource_match"]["pair_audit"],
        exp279_execution_artifact=execution,
    )


def test_exp279_confirmatory_prep_import_contract() -> None:
    from nolane_ai.experiments.exp279_confirmatory_prep import (
        build_exp279_confirmatory_prep,
        validate_exp279_confirmatory_prep,
    )

    assert callable(build_exp279_confirmatory_prep)
    assert callable(validate_exp279_confirmatory_prep)


def test_exp279_blocked_variance_removes_between_stratum_location_shift() -> None:
    from nolane_ai.experiments.exp279_confirmatory_prep import _blocked_relative_effect_summary

    rows = []
    for stratum, hybrid_utility in zip(STRATA, (1.10, 1.30, 1.50), strict=True):
        for replicate in range(2):
            rows.append(
                {
                    "replicate": replicate,
                    "stratum": stratum,
                    "propagation_only": {PRIMARY: 1.0},
                    "hybrid": {PRIMARY: hybrid_utility},
                }
            )
    summary = _blocked_relative_effect_summary(rows, simple_arm="propagation_only")
    assert summary["simple_blocked_mean_utility"] == pytest.approx(1.0)
    assert summary["blocked_within_stratum_sd"] == pytest.approx(0.0, abs=1e-15)
    assert summary["stratum_counts"] == {stratum: 2 for stratum in STRATA}


def test_exp279_confirmatory_prep_freezes_two_contrast_holm_family_before_challenge() -> None:
    from nolane_ai.experiments.exp279_confirmatory_prep import (
        build_exp279_confirmatory_prep,
        validate_exp279_confirmatory_prep,
    )

    execution = _stable_execution()
    prep = build_exp279_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=_registry(execution),
        analysis_code_digest="a" * 64,
    )

    assert prep["schema"] == "NLM-EXP-279-CONFIRMATORY-PREP-V1"
    assert prep["experiment_id"] == "EXP-279"
    assert prep["evidence_level"] == "EV-E2"
    assert prep["decision"] == "UNVERIFIED"
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    assert prep["confirmatory_ready"] is True
    assert prep["confirmatory_data_consumed"] is False
    assert prep["challenge_materialized"] is False
    assert prep["decision_rule_executed"] is False

    frozen = prep["frozen_analysis"]
    assert frozen["primary_endpoint"] == PRIMARY
    assert frozen["primary_direction"] == "higher"
    assert frozen["effect_type"] == "equal_weight_blocked_ratio_of_means_relative_gain"
    assert frozen["primary_contrasts"] == [
        "hybrid_vs_propagation_only",
        "hybrid_vs_branch_only",
    ]
    assert frozen["mesi_relative_gain"] == 0.08
    assert frozen["familywise_alpha"] == 0.05
    assert frozen["planning_alpha"] == 0.025
    assert frozen["holm_step_down_thresholds"] == [0.025, 0.05]
    assert frozen["bootstrap_samples"] == 10_000
    assert frozen["multiplicity_family"] == "PROPAGATION_ROUTING"
    assert frozen["analysis_method"] == (
        "blocked paired contrasts; Holm-adjusted familywise comparisons against best simpler arm"
    )
    assert frozen["structure_fit_strata"] == STRATA
    assert frozen["best_simple_selection_rule"] == (
        "higher equal-weight blocked mean primary utility; exact tie -> propagation_only"
    )
    assert frozen["protected_solution_floor"] == "hybrid >= best_simple - 0.01"
    assert frozen["scientific_denominator_policy"] == (
        "no_epsilon; every simpler-arm blocked mean denominator must be finite and positive"
    )

    freeze = prep["sample_size_freeze"]
    assert freeze["method"] == "max-two-contrast-blocked-within-stratum-normal-approximation-v1"
    assert freeze["power_target"] == 0.90
    assert freeze["min_n"] == 32
    assert freeze["max_n"] == 128
    assert freeze["paired"] is True
    assert freeze["pilot_reuse_as_confirmatory"] is False
    assert freeze["contrast_required_n"] == {
        "hybrid_vs_propagation_only": 1,
        "hybrid_vs_branch_only": 1,
    }
    assert freeze["unclamped_required_n"] == 1
    assert freeze["confirmatory_n"] == 32

    lineage = prep["confirmatory_lineage"]
    assert lineage["lane"] == "POST_FREEZE_CHALLENGE_RESERVED_UNCONSUMED"
    assert lineage["pilot_reuse_forbidden"] is True
    assert lineage["seed_materialization_status"] == "NOT_EXECUTED"
    assert lineage["stratum_schedule"] == STRATA
    assert len(lineage["reserved_replicate_ids"]) == 32
    assert min(lineage["reserved_replicate_ids"]) > max(
        row["replicate"] for row in execution["evaluation"]["per_replicate"]
    )
    assert validate_exp279_confirmatory_prep(prep) == []


def test_exp279_confirmatory_prep_refuses_epsilon_rescue_for_zero_simple_denominator() -> None:
    from nolane_ai.experiments.exp279_paired_runner import _aggregate_rows, _artifact_digest, validate_exp279_paired_development
    from nolane_ai.experiments.exp279_confirmatory_prep import build_exp279_confirmatory_prep

    execution = _stable_execution()
    for row in execution["evaluation"]["per_replicate"]:
        row["propagation_only"]["verified_solution_rate"] = 0.0
        row["propagation_only"][PRIMARY] = 0.0
    execution["evaluation"]["aggregate"] = _aggregate_rows(execution["evaluation"]["per_replicate"])
    execution["artifact_digest"] = _artifact_digest(execution)
    assert validate_exp279_paired_development(execution) == []

    prep = build_exp279_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=_registry(execution),
        analysis_code_digest="a" * 64,
    )
    assert prep["status"] == "NOT_READY_SIMPLE_BLOCKED_MEAN_NONPOSITIVE"
    assert prep["confirmatory_ready"] is False
    assert prep["sample_size_freeze"]["confirmatory_n"] is None
    assert prep["confirmatory_lineage"]["reserved_replicate_ids"] == []


def test_exp279_confirmatory_prep_rejects_development_multiplicity_tamper_even_after_rehash() -> None:
    from nolane_ai.experiments.exp279_paired_runner import _artifact_digest
    from nolane_ai.experiments.exp279_confirmatory_prep import build_exp279_confirmatory_prep

    execution = _stable_execution()
    registry = _registry(execution)
    tampered = deepcopy(execution)
    tampered["multiplicity_family"] = "POST_HOC_ROUTING"
    tampered["artifact_digest"] = _artifact_digest(tampered)
    with pytest.raises(ValueError, match="development artifact"):
        build_exp279_confirmatory_prep(
            experiment=_frozen_experiment(),
            execution_artifact=tampered,
            arm_registry=registry,
            analysis_code_digest="a" * 64,
        )
