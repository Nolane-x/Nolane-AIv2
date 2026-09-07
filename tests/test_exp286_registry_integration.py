from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


def _protocol():
    return {
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "status": "FROZEN_V1",
        "experiments": [
            {
                "experiment_id": "EXP-277",
                "arms": [
                    {"id": "arcs_branch", "description": "V0.15 ARCS recurrent-depth + branch bank + verifier court"},
                    {"id": "oracle_cbrf", "description": "same substrate with ground-truth constraint/factor representation"},
                ],
                "resource_match": {"parameter_budget": "matched active parameter count", "inference_budget": "same max accounted FLOPs per episode", "world_pairing": "same world lineage and replicate index"},
            },
            {
                "experiment_id": "EXP-279",
                "arms": [
                    {"id": "propagation_only", "description": "constraint propagation without branch search"},
                    {"id": "branch_only", "description": "ARCS branch search without CBRF propagation"},
                    {"id": "hybrid", "description": "propagation followed by branch search when residual uncertainty remains"},
                ],
                "resource_match": {"parameter_budget": "reclaimed parameters assigned to simpler rivals", "inference_budget": "equal max accounted FLOPs", "structure_fit": "predeclared strata"},
            },
            {
                "experiment_id": "EXP-282",
                "arms": [
                    {"id": "recurrent_hidden", "description": "matched recurrent state without explicit belief representation"},
                    {"id": "explicit_belief", "description": "explicit calibrated belief state over hidden world variables"},
                ],
                "resource_match": {"parameter_budget": "equal state/controller parameters", "observation_history": "identical", "compute_budget": "matched"},
            },
            {
                "experiment_id": "EXP-286",
                "arms": [
                    {"id": "chronological_failure", "description": "no conflict core; chronological rollback"},
                    {"id": "oracle_conflict_core", "description": "ground-truth conflict core supplied at contradiction"},
                ],
                "resource_match": {"parameter_budget": "oracle metadata is information, not free neural parameters; oracle-info receipt reported", "episode_budget": "same max FLOPs"},
            },
        ],
    }


def _model_audit():
    return audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))


def _execution():
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development

    return run_exp286_paired_development(
        root_seed="exp286-registry-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=3,
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


def test_registry_accepts_exp286_pair_and_execution_but_keeps_match_court_blocked() -> None:
    from nolane_ai.experiments.neural_arm_registry import (
        build_neural_arm_registry,
        validate_neural_arm_registry,
    )

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp286_pair_audit=pair_audit,
        exp286_execution_artifact=execution,
    )
    exp286 = registry["experiments"]["EXP-286"]
    evidence = exp286["paired_execution_evidence"]
    aggregate = execution["evaluation"]["aggregate"]
    assert exp286["development_match_status"] == "PAIRED_CONFLICT_HEADROOM_DEV_READY"
    assert exp286["match_court"] == "BLOCKED"
    assert exp286["arms"]["chronological_failure"]["implementation_status"] == "IMPLEMENTED"
    assert exp286["arms"]["oracle_conflict_core"]["implementation_status"] == "IMPLEMENTED"
    assert exp286["resource_match_evidence"]["parameter_match"] is True
    assert exp286["resource_match_evidence"]["active_functional_parameter_match"] is True
    assert exp286["resource_match_evidence"]["optimizer_visible_parameter_match"] is True
    assert exp286["resource_match_evidence"]["compute_budget_closed"] is True
    assert exp286["resource_match_evidence"]["oracle_information_separation"] is True
    assert evidence["artifact_digest"] == execution["artifact_digest"]
    assert evidence["code_digest"] == execution["code_digest"]
    assert evidence["training_replicates"] == 2
    assert evidence["evaluation_replicates"] == 3
    assert evidence["learned_conflict_localizer_validated"] is False
    assert evidence["descriptive_relative_flop_reduction"] == pytest.approx(
        aggregate["descriptive_relative_flop_reduction"]
    )
    assert evidence["oracle_minus_chronological_verified_solution_rate"] == pytest.approx(
        aggregate["oracle_minus_chronological_verified_solution_rate"]
    )
    assert evidence["chronological_censored_episode_count"] == aggregate[
        "chronological_censored_episode_count"
    ]
    assert evidence["oracle_censored_episode_count"] == aggregate[
        "oracle_censored_episode_count"
    ]
    assert exp286["blockers"] == [
        "EXP-286: confirmatory sample-size/paired-log-cost analysis freeze remains open",
        "EXP-286: confirmatory-open execution remains unrun",
        "EXP-286: post-freeze challenge evidence remains unavailable",
        "EXP-286: learned conflict-core localization remains unvalidated",
    ]
    assert validate_neural_arm_registry(registry) == []


def test_registry_accepts_exp286_pair_only_without_claiming_paired_execution_ready() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp286_pair_audit=pair_audit,
    )
    exp286 = registry["experiments"]["EXP-286"]
    assert exp286["development_match_status"] == "PARAMETER_COMPUTE_ORACLE_SEPARATION_CLOSED"
    assert exp286["match_court"] == "BLOCKED"
    assert "paired_execution_evidence" not in exp286
    assert exp286["blockers"] == [
        "EXP-286: matched arms are not yet integrated into paired conflict-headroom DEVELOPMENT execution lineage"
    ]


def test_registry_rejects_tampered_exp286_pair_or_execution_artifact() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = deepcopy(execution["resource_match"]["pair_audit"])
    bad_pair = deepcopy(pair_audit)
    bad_pair["oracle_information_separation"] = False
    with pytest.raises(ValueError, match="EXP-286 matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp286_pair_audit=bad_pair,
        )

    bad_execution = deepcopy(execution)
    bad_execution["evaluation"]["per_replicate"][0]["future_conflict_core_leakage"] = True
    with pytest.raises(ValueError, match="EXP-286 paired execution artifact"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp286_pair_audit=pair_audit,
            exp286_execution_artifact=bad_execution,
        )


def test_registry_rejects_exp286_execution_without_pair_audit() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    with pytest.raises(ValueError, match="requires matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp286_execution_artifact=_execution(),
        )


def test_registry_rejects_exp286_execution_with_wrong_protocol_digest() -> None:
    from nolane_ai.experiments.exp286_paired_runner import _artifact_digest
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    wrong_protocol = deepcopy(execution)
    wrong_protocol["protocol_digest"] = "q" * 64
    wrong_protocol["artifact_digest"] = _artifact_digest(wrong_protocol)
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp286_pair_audit=pair_audit,
            exp286_execution_artifact=wrong_protocol,
        )


def test_registry_rejects_exp286_execution_claiming_learned_localizer_validation() -> None:
    from nolane_ai.experiments.exp286_paired_runner import _artifact_digest
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    overclaim = deepcopy(execution)
    overclaim["learned_conflict_localizer_validated"] = True
    overclaim["artifact_digest"] = _artifact_digest(overclaim)
    with pytest.raises(ValueError, match="learned conflict localizer"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp286_pair_audit=pair_audit,
            exp286_execution_artifact=overclaim,
        )


def test_registry_validator_rejects_manual_exp286_localizer_promotion_after_rehash() -> None:
    from nolane_ai.experiments.neural_arm_registry import (
        _digest,
        build_neural_arm_registry,
        validate_neural_arm_registry,
    )

    execution = _execution()
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp286_pair_audit=execution["resource_match"]["pair_audit"],
        exp286_execution_artifact=execution,
    )
    registry["experiments"]["EXP-286"]["paired_execution_evidence"][
        "learned_conflict_localizer_validated"
    ] = True
    registry["registry_digest"] = _digest(registry)
    errors = validate_neural_arm_registry(registry)
    assert "EXP-286 development evidence cannot validate a learned conflict localizer" in errors
