from __future__ import annotations

from copy import deepcopy

import pytest


def _protocol_exp277() -> dict:
    return {
        "experiment_id": "EXP-277",
        "primary_endpoint": {"metric": "verified_utility_per_accounted_flop", "direction": "higher"},
        "protected_endpoints": [
            {"metric": "verified_solution_rate", "floor": "oracle_cbrf >= arcs_branch - 0.005"},
            {"metric": "wall_energy_per_episode", "floor": "report-only at Stage-A; cannot be hidden from cost ledger"},
        ],
        "mesi": {"type": "relative_gain", "value": 0.10, "unit": "fraction"},
        "sample_size_plan": {"power_target": 0.90, "min_n": 32, "max_n": 128, "paired": True},
        "analysis_method": "paired effect with bootstrap confidence interval; report mean, median, sign consistency and divergence rate",
        "multiplicity_family": "CBRF_HEADROOM",
        "failure_policy": {"scientific_failure_kept": True, "rerun_only": "predeclared infrastructure failure before first model update/inference step"},
        "challenge_generator": {"lane": "POST_FREEZE_CHALLENGE", "seed_rule": "SHA256(protocol_digest|beacon|experiment|stream|replicate)"},
    }


def _execution(arcs_utilities, oracle_utilities=None, *, solution_delta=0.0) -> dict:
    if oracle_utilities is None:
        oracle_utilities = [u * 1.10 for u in arcs_utilities]
    assert len(arcs_utilities) == len(oracle_utilities)
    rows = []
    for idx, (arcs_u, oracle_u) in enumerate(zip(arcs_utilities, oracle_utilities, strict=True)):
        arcs_solution = 0.75
        oracle_solution = arcs_solution + solution_delta
        rows.append(
            {
                "replicate": 10_000 + idx,
                "paired_batch_digest": f"batch-{idx}",
                "world_pairing_closed": True,
                "arcs_branch": {
                    "verified_solution_rate": arcs_solution,
                    "verified_decision_accuracy": 0.75,
                    "accounted_flops_per_episode": 1000,
                    "verified_utility_per_accounted_flop": float(arcs_u),
                },
                "oracle_cbrf": {
                    "verified_solution_rate": oracle_solution,
                    "verified_decision_accuracy": 0.75,
                    "accounted_flops_per_episode": 1000,
                    "verified_utility_per_accounted_flop": float(oracle_u),
                },
                "oracle_relative_verified_utility_gain": 0.0,
            }
        )
    return {
        "schema": "NLM-EXP-277-PAIRED-DEV-EVAL-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "protocol_digest": "p" * 64,
        "code_digest": "c" * 64,
        "artifact_digest": "a" * 64,
        "final_state": {"arcs_branch_digest": "h" * 64, "oracle_cbrf_digest": "o" * 64},
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "same_world_lineage": True,
            "compute_budget_closed": True,
            "declared_max_accounted_flops_per_episode": 1000,
            "pair_audit": {
                "schema": "NLM-EXP-277-MATCHED-ARMS-DEV-V1",
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "parameter_match": True,
                "functional_parameter_match": True,
                "oracle_information_separation": True,
                "compute_budget_closed": True,
            },
            "pair_audit_digest": "q" * 64,
        },
        "oracle_information_receipt": {
            "artifact": "oracle_incidence",
            "ground_truth": True,
            "delivered_to": ["oracle_cbrf"],
            "withheld_from": ["arcs_branch"],
            "arcs_received_oracle_incidence": False,
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": 16,
            "paired_batch_digests": [f"train-{i}" for i in range(16)],
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": 10_000,
            "replicates": len(rows),
            "per_replicate": rows,
        },
    }


def _registry() -> dict:
    return {
        "schema": "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_digest": "p" * 64,
        "registry_digest": "r" * 64,
        "experiments": {
            "EXP-277": {
                "development_match_status": "PAIRED_STRUCTURE_DENSE_DEV_READY",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {"artifact_digest": "a" * 64, "code_digest": "c" * 64},
            }
        },
    }


def test_exp277_confirmatory_prep_requires_32_paired_pilot_replicates() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    with pytest.raises(ValueError, match="at least 32 paired pilot replicates"):
        build_exp277_confirmatory_prep(
            experiment=_protocol_exp277(),
            execution_artifact=_execution([1e-6] * 31),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )


def test_exp277_confirmatory_prep_freezes_relative_effect_and_feasible_n() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import (
        build_exp277_confirmatory_prep,
        validate_exp277_confirmatory_prep,
    )

    arcs = [1.0e-6 + (i % 4) * 0.05e-6 for i in range(32)]
    oracle = [a * (1.05 + (i % 5) * 0.025) for i, a in enumerate(arcs)]
    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution(arcs, oracle),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["schema"] == "NLM-EXP-277-CONFIRMATORY-PREP-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["decision_rule_executed"] is False
    assert payload["frozen_analysis"]["primary_endpoint"] == "verified_utility_per_accounted_flop"
    assert payload["frozen_analysis"]["effect_type"] == "ratio_of_means_relative_gain"
    assert payload["frozen_analysis"]["mesi_relative_gain"] == pytest.approx(0.10)
    assert payload["frozen_analysis"]["endpoint_alpha"] == pytest.approx(0.025)
    assert payload["frozen_analysis"]["bootstrap_samples"] == 10_000
    assert payload["pilot_summary"]["baseline_mean_utility"] > 0.0
    assert payload["pilot_summary"]["paired_sd"] >= 0.0
    assert payload["pilot_summary"]["paired_effect_digest"]
    assert 32 <= payload["sample_size_freeze"]["confirmatory_n"] <= 128
    assert payload["confirmatory_lineage"]["seed_materialization_status"] == "NOT_EXECUTED"
    assert payload["confirmatory_lineage"]["pilot_reuse_forbidden"] is True
    assert validate_exp277_confirmatory_prep(payload) == []


def test_exp277_confirmatory_prep_fails_closed_when_baseline_mean_nonpositive() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution([0.0] * 32, [1e-6] * 32),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["status"] == "NOT_READY_PRIMARY_BASELINE_NONPOSITIVE"
    assert payload["sample_size_freeze"]["confirmatory_n"] is None
    assert payload["confirmatory_lineage"]["reserved_replicate_ids"] == []
    assert "baseline" in " ".join(payload["remaining_blockers"]).lower()


def test_exp277_confirmatory_prep_refuses_to_truncate_n_above_128() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    arcs = [1.0] * 32
    oracle = [0.0 if i % 2 == 0 else 2.0 for i in range(32)]
    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution(arcs, oracle),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["status"] == "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
    assert payload["sample_size_freeze"]["unclamped_required_n"] > 128
    assert payload["sample_size_freeze"]["confirmatory_n"] is None
    assert payload["confirmatory_lineage"]["reserved_replicate_ids"] == []


def test_exp277_confirmatory_prep_reserves_ids_disjoint_from_training_and_pilot() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution([1.0] * 32, [1.1] * 32),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    train_ids = set(payload["pilot_summary"]["training_replicate_ids"])
    pilot_ids = set(payload["pilot_summary"]["replicate_ids"])
    reserved = payload["confirmatory_lineage"]["reserved_replicate_ids"]
    assert train_ids.isdisjoint(reserved)
    assert pilot_ids.isdisjoint(reserved)
    assert reserved == list(range(max(pilot_ids | train_ids) + 1, max(pilot_ids | train_ids) + 1 + len(reserved)))


def test_exp277_confirmatory_prep_rejects_registry_mismatch_and_non_evaluation_lane() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    registry = _registry()
    registry["experiments"]["EXP-277"]["paired_execution_evidence"]["artifact_digest"] = "x" * 64
    with pytest.raises(ValueError, match="execution artifact digest"):
        build_exp277_confirmatory_prep(
            experiment=_protocol_exp277(),
            execution_artifact=_execution([1.0] * 32),
            arm_registry=registry,
            analysis_code_digest="d" * 64,
        )

    execution = _execution([1.0] * 32)
    execution["evaluation"]["rng_stream"] = "challenge"
    with pytest.raises(ValueError, match="evaluation RNG"):
        build_exp277_confirmatory_prep(
            experiment=_protocol_exp277(),
            execution_artifact=execution,
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )


def test_exp277_confirmatory_prep_rejects_protocol_and_analysis_drift() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    protocol = _protocol_exp277()
    protocol["mesi"]["value"] = 0.01
    with pytest.raises(ValueError, match="MESI drift"):
        build_exp277_confirmatory_prep(
            experiment=protocol,
            execution_artifact=_execution([1.0] * 32),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )

    with pytest.raises(ValueError, match="alpha drift"):
        build_exp277_confirmatory_prep(
            experiment=_protocol_exp277(),
            execution_artifact=_execution([1.0] * 32),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
            familywise_alpha=0.10,
        )


def test_exp277_confirmatory_prep_validator_rejects_rehashed_semantic_tamper() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import (
        _prep_digest,
        build_exp277_confirmatory_prep,
        validate_exp277_confirmatory_prep,
    )

    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution([1.0] * 32, [1.1] * 32),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    bad = deepcopy(payload)
    bad["frozen_analysis"]["endpoint_alpha"] = 0.05
    bad["frozen_analysis"]["effect_type"] = "mean_of_per_replicate_ratios"
    bad["sample_size_freeze"]["min_n"] = 1
    bad["sample_size_freeze"]["pilot_reuse_as_confirmatory"] = True
    bad["confirmatory_lineage"]["seed_materialization_status"] = "EXECUTED"
    bad["decision"] = "PROMOTE_TO_NEXT_STAGE"
    bad["prep_digest"] = _prep_digest(bad)
    errors = validate_exp277_confirmatory_prep(bad)
    assert "confirmatory prep cannot promote H-CBRF-01" in errors
    assert "EXP-277 endpoint alpha drift" in errors
    assert "EXP-277 primary effect contract drift" in errors
    assert "EXP-277 frozen sample-size bounds drift" in errors
    assert "confirmatory prep must forbid pilot reuse as confirmatory" in errors
    assert "confirmatory seed materialization must remain NOT_EXECUTED during prep" in errors


def test_exp277_confirmatory_prep_validator_recomputes_required_n_after_rehash() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import (
        _prep_digest,
        build_exp277_confirmatory_prep,
        validate_exp277_confirmatory_prep,
    )

    payload = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=_execution([1.0] * 32, [1.08, 1.12, 1.10, 1.14] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    payload["sample_size_freeze"]["unclamped_required_n"] += 1
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp277_confirmatory_prep(payload)
    assert "unclamped required n does not match frozen power formula" in errors
