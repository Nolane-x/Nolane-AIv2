from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
PRIMARY_CONTRASTS = [
    "hybrid_vs_propagation_only",
    "hybrid_vs_branch_only",
]


def _machinery() -> dict[str, str]:
    return {
        "challenge_generator_digest": "1" * 64,
        "beacon_seed_derivation_digest": "2" * 64,
        "executor_code_digest": "3" * 64,
        "reconstruction_code_digest": "4" * 64,
    }


def _build(tmp_path):
    from nolane_ai.experiments.exp279_checkpoint import build_exp279_trained_checkpoint
    from nolane_ai.experiments.exp279_confirmatory_authorization import (
        build_exp279_gate_a_authorization,
        build_exp279_gate_a_seal,
    )
    from nolane_ai.experiments.exp279_confirmatory_prep import build_exp279_confirmatory_prep
    from tests.test_exp279_confirmatory_prep import (
        _frozen_experiment,
        _registry,
        _stable_execution,
    )

    execution = _stable_execution()
    registry = _registry(execution)
    prep = build_exp279_confirmatory_prep(
        experiment=_frozen_experiment(),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest="a" * 64,
    )
    checkpoint = build_exp279_trained_checkpoint(
        execution_artifact=execution,
        checkpoint_path=tmp_path / "exp279-checkpoint.pt",
    )
    authorization = build_exp279_gate_a_authorization(
        prep_artifact=prep,
        checkpoint_receipt=checkpoint,
        development_execution_artifact=execution,
        arm_registry=registry,
        source_tree_digest="s" * 64,
        freeze_commit_sha="f" * 40,
        freeze_commit_timestamp_utc="2026-09-09T10:00:00Z",
        checkpoint_seal_created_at_utc="2026-09-09T10:00:30Z",
        machinery_digests=_machinery(),
    )
    seal = build_exp279_gate_a_seal(authorization=authorization)
    return execution, registry, prep, checkpoint, authorization, seal


def test_exp279_gate_a_authorization_and_seal_freeze_without_materializing_challenge(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_authorization import (
        validate_exp279_gate_a_authorization,
        validate_exp279_gate_a_seal,
    )

    _, _, prep, checkpoint, authorization, seal = _build(tmp_path)

    assert authorization["schema"] == "NLM-EXP-279-CONFIRMATORY-GATE-A-AUTH-V1"
    assert authorization["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_ready"] is False
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["challenge_materialized"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["decision_rule_executed"] is False
    assert authorization["confirmatory_n"] == prep["sample_size_freeze"]["confirmatory_n"]
    assert authorization["reserved_replicate_ids"] == prep["confirmatory_lineage"]["reserved_replicate_ids"]
    assert authorization["stratum_schedule"] == STRATA
    assert authorization["checkpoint"]["receipt_digest"] == checkpoint["receipt_digest"]
    assert validate_exp279_gate_a_authorization(authorization) == []

    assert seal["schema"] == "NLM-EXP-279-CONFIRMATORY-GATE-A-SEAL-V1"
    assert seal["status"] == "CONFIRMATORY_GATE_A_SEALED"
    assert seal["readiness"] == "FROZEN_MACHINERY_AND_CHECKPOINT_READY_FOR_FUTURE_BEACON_ONLY"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["decision_rule_executed"] is False
    assert validate_exp279_gate_a_seal(seal) == []

    rendered = repr(seal).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered
    assert "challenge_batch" not in rendered


def test_exp279_gate_a_binds_holm_routing_strata_checkpoint_and_path_dependent_compute(tmp_path) -> None:
    from nolane_ai.protocol.evidence import canonical_sha256

    execution, _, prep, checkpoint, authorization, seal = _build(tmp_path)
    frozen = prep["frozen_analysis"]
    pair_audit = execution["resource_match"]["pair_audit"]
    ledger = pair_audit["compute_ledger"]

    assert authorization["protocol_digest"] == execution["protocol_digest"]
    assert authorization["source_tree_digest"] == "s" * 64
    assert authorization["development_execution_digest"] == execution["artifact_digest"]
    assert authorization["prep_digest"] == prep["prep_digest"]
    assert authorization["frozen_analysis_digest"] == canonical_sha256(frozen)
    assert authorization["sample_size_freeze_digest"] == canonical_sha256(prep["sample_size_freeze"])
    assert authorization["analysis_code_digest"] == "a" * 64

    assert authorization["primary_contrasts"] == PRIMARY_CONTRASTS
    assert authorization["familywise_alpha"] == pytest.approx(0.05)
    assert authorization["planning_alpha"] == pytest.approx(0.025)
    assert authorization["holm_step_down_thresholds"] == [0.025, 0.05]
    assert authorization["bootstrap_samples"] == 10_000
    assert authorization["pilot_best_simple_not_carried_into_confirmatory"] is True
    assert authorization["stratum_schedule"] == STRATA
    assert authorization["route_threshold"] == pytest.approx(0.5)

    checkpoint_binding = authorization["checkpoint"]
    assert checkpoint_binding["receipt_digest"] == checkpoint["receipt_digest"]
    assert checkpoint_binding["scientific_identity_digest"] == checkpoint["scientific_identity_digest"]
    assert checkpoint_binding["replay_contract_digest"] == checkpoint["replay_contract_digest"]
    assert checkpoint_binding["final_state_digest"] == checkpoint["final_state_digest"]
    assert checkpoint_binding["final_state"] == checkpoint["final_state"]
    assert checkpoint_binding["checkpoint_file_sha256"] == checkpoint["checkpoint_file_sha256"]
    assert checkpoint_binding["state_policy"] == "functional-only"

    resource = authorization["resource_court"]
    for field in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "same_world_lineage",
        "compute_budget_closed",
    ):
        assert resource[field] is True
    assert resource["structure_fit_strata"] == STRATA
    assert resource["pair_audit_digest"] == canonical_sha256(pair_audit)
    assert resource["route_threshold"] == pytest.approx(0.5)
    assert resource["accounted_flops_contract"] == {
        "propagation_only": {
            "mode": "fixed",
            "accounted_flops_per_episode": ledger["propagation_only"]["accounted_flops_per_episode"],
        },
        "branch_only": {
            "mode": "fixed",
            "accounted_flops_per_episode": ledger["branch_only"]["accounted_flops_per_episode"],
        },
        "hybrid": {
            "mode": "route_dependent",
            "stop_accounted_flops_per_episode": ledger["hybrid"]["stop_accounted_flops_per_episode"],
            "branch_accounted_flops_per_episode": ledger["hybrid"]["branch_accounted_flops_per_episode"],
        },
    }
    assert authorization["information_separation"] == {
        "artifact": "constraint_variable_incidence",
        "ground_truth": True,
        "delivered_to": ["propagation_only", "hybrid"],
        "withheld_from": ["branch_only"],
        "branch_only_received_incidence": False,
    }
    assert authorization["machinery_digests"] == _machinery()
    assert seal["authorization_digest"] == authorization["authorization_digest"]


def test_exp279_gate_a_rejects_checkpoint_development_resource_and_timestamp_drift(tmp_path) -> None:
    from nolane_ai.experiments.exp279_checkpoint import _receipt_digest
    from nolane_ai.experiments.exp279_confirmatory_authorization import build_exp279_gate_a_authorization

    execution, registry, prep, checkpoint, _, _ = _build(tmp_path)

    bad_checkpoint = deepcopy(checkpoint)
    bad_checkpoint["final_state"]["hybrid_digest"] = "9" * 64
    bad_checkpoint["receipt_digest"] = _receipt_digest(bad_checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        build_exp279_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=bad_checkpoint,
            development_execution_artifact=execution,
            arm_registry=registry,
            source_tree_digest="s" * 64,
            freeze_commit_sha="f" * 40,
            freeze_commit_timestamp_utc="2026-09-09T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-09T10:00:30Z",
            machinery_digests=_machinery(),
        )

    drift = deepcopy(execution)
    drift["route_config"]["threshold"] = 0.25
    with pytest.raises(ValueError, match="development|route|prep"):
        build_exp279_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=checkpoint,
            development_execution_artifact=drift,
            arm_registry=registry,
            source_tree_digest="s" * 64,
            freeze_commit_sha="f" * 40,
            freeze_commit_timestamp_utc="2026-09-09T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-09T10:00:30Z",
            machinery_digests=_machinery(),
        )

    resource_open = deepcopy(execution)
    resource_open["resource_match"]["compute_budget_closed"] = False
    with pytest.raises(ValueError, match="resource|development|prep"):
        build_exp279_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=checkpoint,
            development_execution_artifact=resource_open,
            arm_registry=registry,
            source_tree_digest="s" * 64,
            freeze_commit_sha="f" * 40,
            freeze_commit_timestamp_utc="2026-09-09T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-09T10:00:30Z",
            machinery_digests=_machinery(),
        )

    with pytest.raises(ValueError, match="strictly after|timestamp"):
        build_exp279_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=checkpoint,
            development_execution_artifact=execution,
            arm_registry=registry,
            source_tree_digest="s" * 64,
            freeze_commit_sha="f" * 40,
            freeze_commit_timestamp_utc="2026-09-09T10:00:30Z",
            checkpoint_seal_created_at_utc="2026-09-09T10:00:00Z",
            machinery_digests=_machinery(),
        )


def test_exp279_gate_a_validators_reject_rehashed_semantic_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_authorization import (
        _authorization_binding,
        _authorization_digest,
        _seal_binding,
        _seal_digest,
        validate_exp279_gate_a_authorization,
        validate_exp279_gate_a_seal,
    )

    _, _, _, _, authorization, seal = _build(tmp_path)

    tampered = deepcopy(authorization)
    tampered["familywise_alpha"] = 0.10
    tampered["binding_digest"] = _authorization_binding(tampered)
    tampered["authorization_digest"] = _authorization_digest(tampered)
    assert any("familywise" in error.lower() or "alpha" in error.lower() for error in validate_exp279_gate_a_authorization(tampered))

    tampered = deepcopy(authorization)
    tampered["pilot_best_simple_not_carried_into_confirmatory"] = False
    tampered["binding_digest"] = _authorization_binding(tampered)
    tampered["authorization_digest"] = _authorization_digest(tampered)
    assert any("pilot" in error.lower() or "best-simple" in error.lower() for error in validate_exp279_gate_a_authorization(tampered))

    bad_seal = deepcopy(seal)
    bad_seal["stratum_schedule"] = list(reversed(STRATA))
    bad_seal["seal_binding_digest"] = _seal_binding(bad_seal)
    bad_seal["seal_digest"] = _seal_digest(bad_seal)
    assert any("stratum" in error.lower() for error in validate_exp279_gate_a_seal(bad_seal))
