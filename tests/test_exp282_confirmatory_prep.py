import pytest


def _protocol_exp282():
    return {
        "experiment_id": "EXP-282",
        "primary_endpoint": {"metric": "grounded_decision_accuracy", "direction": "higher"},
        "protected_endpoints": [
            {"metric": "brier_score", "floor": "explicit_belief <= recurrent_hidden + 0.02"},
            {"metric": "accounted_flops", "floor": "difference <= 0.05 relative unless included in primary cost normalization"},
        ],
        "mesi": {"type": "absolute_gain", "value": 0.03, "unit": "accuracy_fraction"},
        "sample_size_plan": {"power_target": 0.90, "min_n": 32, "max_n": 128, "paired": True},
        "analysis_method": "paired accuracy difference with bootstrap CI plus calibration guard",
        "multiplicity_family": "BELIEF_STATE",
    }


def _execution(effects, *, brier=0.0):
    raw = []
    for idx, effect in enumerate(effects):
        raw.append({
            "replicate": 1000 + idx,
            "batch_digest": f"batch-{idx}",
            "recurrent_hidden": {"grounded_decision_accuracy": 0.5, "brier_score": 0.25},
            "explicit_belief": {"grounded_decision_accuracy": 0.5 + effect, "brier_score": 0.25 + brier},
            "explicit_minus_recurrent_accuracy": effect,
            "explicit_minus_recurrent_brier": brier,
        })
    return {
        "schema": "NLM-EXP-282-PAIRED-DEV-EVAL-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_ready": False,
        "protocol_digest": "p" * 64,
        "code_digest": "c" * 64,
        "artifact_digest": "a" * 64,
        "checkpoint": {"schema": "NLM-EXP-282-PAIRED-CHECKPOINT-V1", "checkpoint_sha256": "k" * 64, "state_policy": "functional-only"},
        "resource_match": {"parameter_match": True, "observation_history_match": True, "accounted_flop_match": True, "relative_accounted_flop_difference": 0.0},
        "training": {"rng_stream": "augmentation", "start_replicate": 0, "replicates": 3, "batch_digests": ["train-0", "train-1", "train-2"]},
        "evaluation": {"rng_stream": "evaluation", "start_replicate": 1000, "replicates": len(raw), "per_replicate": raw},
    }


def _registry():
    return {
        "schema": "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "protocol_digest": "p" * 64,
        "registry_digest": "r" * 64,
        "experiments": {
            "EXP-282": {
                "development_match_status": "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {"artifact_digest": "a" * 64},
            }
        },
    }


def test_confirmatory_prep_requires_full_32_replicate_pilot():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    with pytest.raises(ValueError, match="at least 32 paired pilot replicates"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=_execution([0.02] * 31),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )


def test_confirmatory_prep_freezes_numeric_analysis_and_sample_size_when_power_is_feasible():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep, validate_exp282_confirmatory_prep

    effects = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07] * 4
    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution(effects, brier=-0.01),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["schema"] == "NLM-EXP-282-CONFIRMATORY-OPEN-PREP-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["status"] == "CONFIRMATORY_OPEN_PREPARED"
    assert payload["confirmatory_data_consumed"] is False
    assert payload["frozen_analysis"]["primary_endpoint"] == "grounded_decision_accuracy"
    assert payload["frozen_analysis"]["mesi_absolute_gain"] == pytest.approx(0.03)
    assert payload["frozen_analysis"]["power_target"] == pytest.approx(0.90)
    assert payload["frozen_analysis"]["familywise_alpha"] == pytest.approx(0.05)
    assert 32 <= payload["sample_size_freeze"]["confirmatory_n"] <= 128
    assert payload["sample_size_freeze"]["method"] == "paired-normal-approximation-from-pilot-sd-v1"
    assert payload["pilot_summary"]["n"] == 32
    assert payload["pilot_summary"]["paired_sd"] > 0
    assert payload["lineage"]["paired_checkpoint_sha256"] == "k" * 64
    assert payload["remaining_blockers"] == [
        "confirmatory-open execution has not consumed any reserved confirmatory data",
        "post-freeze challenge beacon and independent replication remain open",
    ]
    assert validate_exp282_confirmatory_prep(payload) == []


def test_confirmatory_prep_refuses_to_truncate_required_sample_size_above_protocol_maximum():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    effects = [-0.6, 0.6] * 16
    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution(effects),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["status"] == "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
    assert payload["sample_size_freeze"]["confirmatory_n"] is None
    assert payload["sample_size_freeze"]["unclamped_required_n"] > 128
    assert "pilot variance implies required n above frozen maximum 128" in payload["remaining_blockers"]


def test_confirmatory_prep_rejects_registry_execution_mismatch_and_tamper():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep, validate_exp282_confirmatory_prep

    registry = _registry()
    registry["experiments"]["EXP-282"]["paired_execution_evidence"]["artifact_digest"] = "x" * 64
    with pytest.raises(ValueError, match="execution artifact digest"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=_execution([0.02] * 32),
            arm_registry=registry,
            analysis_code_digest="d" * 64,
        )

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    payload["decision"] = "PROMOTE_TO_NEXT_STAGE"
    payload["frozen_analysis"]["mesi_absolute_gain"] = 0.0
    errors = validate_exp282_confirmatory_prep(payload)
    assert "confirmatory prep cannot promote a neural claim" in errors
    assert "EXP-282 MESI drift" in errors
    assert "confirmatory prep digest mismatch" in errors


def test_confirmatory_prep_rejects_alpha_drift_and_non_evaluation_pilot_lane():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    with pytest.raises(ValueError, match="alpha drift"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=_execution([0.02] * 32),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
            familywise_alpha=0.10,
        )

    execution = _execution([0.02] * 32)
    execution["evaluation"]["rng_stream"] = "challenge"
    with pytest.raises(ValueError, match="evaluation RNG"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=execution,
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )


def test_confirmatory_prep_reserves_new_replicates_and_rejects_sample_size_tamper():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep, validate_exp282_confirmatory_prep

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    pilot_ids = set(payload["pilot_summary"]["replicate_ids"])
    confirmatory_ids = set(payload["confirmatory_lineage"]["reserved_replicate_ids"])
    assert pilot_ids.isdisjoint(confirmatory_ids)
    assert len(confirmatory_ids) == payload["sample_size_freeze"]["confirmatory_n"]
    assert payload["confirmatory_lineage"]["pilot_reuse_forbidden"] is True

    payload["sample_size_freeze"]["confirmatory_n"] += 1
    errors = validate_exp282_confirmatory_prep(payload)
    assert "confirmatory n does not match frozen sample-size rule" in errors
    assert "confirmatory prep digest mismatch" in errors


def test_confirmatory_prep_rejects_internally_inconsistent_paired_contrasts():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    execution = _execution([0.02] * 32, brier=-0.01)
    execution["evaluation"]["per_replicate"][0]["explicit_minus_recurrent_accuracy"] = 0.5
    with pytest.raises(ValueError, match="accuracy contrast mismatch"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=execution,
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )


def test_confirmatory_prep_exposes_reproducible_power_constants_and_effect_digest():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    constants = payload["sample_size_freeze"]["planning_constants"]
    assert constants["alpha_tail"] == "one_sided_lower_bound"
    assert constants["z_alpha"] > 0
    assert constants["z_power"] > 0
    assert len(payload["pilot_summary"]["paired_effect_digest"]) == 64


def test_confirmatory_prep_validator_recomputes_required_n_even_after_rehash():
    from nolane_ai.experiments.exp282_confirmatory_prep import (
        _prep_digest,
        build_exp282_confirmatory_prep,
        validate_exp282_confirmatory_prep,
    )

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    payload["sample_size_freeze"]["unclamped_required_n"] += 1
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp282_confirmatory_prep(payload)
    assert "unclamped required n does not match frozen power formula" in errors


def test_confirmatory_prep_validator_rejects_semantic_contract_drift_even_after_rehash():
    from nolane_ai.experiments.exp282_confirmatory_prep import (
        _prep_digest,
        build_exp282_confirmatory_prep,
        validate_exp282_confirmatory_prep,
    )

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    payload["frozen_analysis"]["primary_direction"] = "lower"
    payload["frozen_analysis"]["inference_tail"] = "two_sided"
    payload["pilot_summary"]["source_lane"] = "CONFIRMATORY_OPEN"
    payload["sample_size_freeze"]["min_n"] = 1
    payload["sample_size_freeze"]["max_n"] = 999
    payload["sample_size_freeze"]["paired"] = False
    payload["sample_size_freeze"]["pilot_reuse_as_confirmatory"] = True
    payload["confirmatory_lineage"]["lane"] = "DEVELOPMENT_PILOT_ONLY"
    payload["confirmatory_lineage"]["seed_materialization_status"] = "EXECUTED"
    payload["prep_digest"] = _prep_digest(payload)
    errors = validate_exp282_confirmatory_prep(payload)
    assert "EXP-282 primary direction drift" in errors
    assert "EXP-282 inference-tail drift" in errors
    assert "EXP-282 pilot source lane drift" in errors
    assert "EXP-282 frozen sample-size bounds drift" in errors
    assert "EXP-282 paired sample-size contract drift" in errors
    assert "confirmatory prep must forbid pilot reuse as confirmatory" in errors
    assert "EXP-282 confirmatory lineage lane drift" in errors
    assert "confirmatory seed materialization must remain NOT_EXECUTED during prep" in errors


def test_confirmatory_prep_rejects_training_evaluation_world_overlap_and_binds_training_lineage():
    from nolane_ai.experiments.exp282_confirmatory_prep import build_exp282_confirmatory_prep

    execution = _execution([0.02] * 32)
    execution["training"]["replicates"] = 1001
    execution["training"]["batch_digests"] = [f"train-{i}" for i in range(1001)]
    with pytest.raises(ValueError, match="training and evaluation replicate lineages overlap"):
        build_exp282_confirmatory_prep(
            experiment=_protocol_exp282(),
            execution_artifact=execution,
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )

    payload = build_exp282_confirmatory_prep(
        experiment=_protocol_exp282(),
        execution_artifact=_execution([0.01, 0.02, 0.03, 0.04] * 8),
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    assert payload["pilot_summary"]["training_replicate_ids"] == [0, 1, 2]
    assert set(payload["confirmatory_lineage"]["reserved_replicate_ids"]).isdisjoint(
        payload["pilot_summary"]["training_replicate_ids"]
    )
