from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from tests.test_neural_arm_registry import _audit, _protocol_subset


def _exp279_triplet_audit_and_execution():
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development
    from nolane_ai.experiments.matched_routing_arms import (
        audit_matched_exp279_arm_triplet,
        build_matched_exp279_arm_triplet,
    )

    propagation, branch, hybrid = build_matched_exp279_arm_triplet(
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=3,
        variables=4,
        constraints=2,
    )
    execution = run_exp279_paired_development(
        root_seed="registry-exp279",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=3,
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
    return pair_audit, execution


def test_registry_accepts_exp279_pair_audit_but_keeps_confirmatory_gate_closed() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry

    pair_audit, _ = _exp279_triplet_audit_and_execution()
    artifact = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest="p" * 64,
        model_audit=_audit(),
        exp279_pair_audit=pair_audit,
    )
    exp279 = artifact["experiments"]["EXP-279"]
    assert exp279["development_match_status"] == "PARAMETER_RECLAIM_COMPUTE_STRATA_CLOSED"
    evidence = exp279["resource_match_evidence"]
    assert evidence["parameter_match"] is True
    assert evidence["functional_parameter_match"] is True
    assert evidence["active_functional_parameter_match"] is True
    assert evidence["reclaimed_parameter_assignment_closed"] is True
    assert evidence["compute_budget_closed"] is True
    assert evidence["structure_fit_strata"] == ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
    for arm in ("propagation_only", "branch_only", "hybrid"):
        assert exp279["arms"][arm]["implementation_status"] == "IMPLEMENTED"
        assert exp279["arms"][arm]["implementation_tier"] == "MATCHED_EXPERIMENT_LOCAL_NEURAL_ARM"
    assert exp279["match_court"] == "BLOCKED"
    assert exp279["blockers"] == [
        "EXP-279: matched arms are not yet integrated into paired structure-fit evaluator lineage"
    ]
    assert validate_neural_arm_registry(artifact) == []


def test_registry_accepts_exp279_paired_execution_without_false_promotion() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry, validate_neural_arm_registry

    pair_audit, execution = _exp279_triplet_audit_and_execution()
    artifact = build_neural_arm_registry(
        protocol=_protocol_subset(),
        protocol_digest="p" * 64,
        model_audit=_audit(),
        exp279_pair_audit=pair_audit,
        exp279_execution_artifact=execution,
    )
    exp279 = artifact["experiments"]["EXP-279"]
    evidence = exp279["paired_execution_evidence"]
    assert exp279["development_match_status"] == "PAIRED_ROUTING_DEV_READY"
    assert evidence["artifact_digest"] == execution["artifact_digest"]
    assert evidence["code_digest"] == execution["code_digest"]
    assert evidence["training_replicates"] == 3
    assert evidence["evaluation_replicates"] == 3
    assert evidence["best_simple_arm"] in {"propagation_only", "branch_only"}
    assert evidence["hybrid_relative_utility_gain"] == pytest.approx(
        execution["evaluation"]["aggregate"]["hybrid_relative_utility_gain"]
    )
    assert evidence["hybrid_minus_best_simple_verified_solution_rate"] == pytest.approx(
        execution["evaluation"]["aggregate"]["hybrid_minus_best_simple_verified_solution_rate"]
    )
    assert exp279["match_court"] == "BLOCKED"
    assert exp279["blockers"] == [
        "EXP-279: confirmatory sample-size/blocked-analysis freeze, confirmatory-open execution and post-freeze challenge evidence remain open"
    ]
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert validate_neural_arm_registry(artifact) == []


def test_registry_rejects_invalid_exp279_pair_audit_and_unpaired_execution() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    pair_audit, execution = _exp279_triplet_audit_and_execution()
    bad = deepcopy(pair_audit)
    bad["compute_budget_closed"] = False
    with pytest.raises(ValueError, match="EXP-279 matched triplet audit"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp279_pair_audit=bad,
        )
    with pytest.raises(ValueError, match="requires matched triplet audit"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp279_execution_artifact=execution,
        )


def test_registry_rejects_tampered_or_wrong_protocol_exp279_execution() -> None:
    from nolane_ai.experiments.exp279_paired_runner import _artifact_digest
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    pair_audit, execution = _exp279_triplet_audit_and_execution()
    tampered = deepcopy(execution)
    tampered["evaluation"]["per_replicate"].pop()
    with pytest.raises(ValueError, match="EXP-279 paired execution artifact"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp279_pair_audit=pair_audit,
            exp279_execution_artifact=tampered,
        )

    wrong_protocol = deepcopy(execution)
    wrong_protocol["protocol_digest"] = "q" * 64
    wrong_protocol["artifact_digest"] = _artifact_digest(wrong_protocol)
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        build_neural_arm_registry(
            protocol=_protocol_subset(),
            protocol_digest="p" * 64,
            model_audit=_audit(),
            exp279_pair_audit=pair_audit,
            exp279_execution_artifact=wrong_protocol,
        )
