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
        "challenge_generator": {"lane": "POST_FREEZE_CHALLENGE", "seed_rule": "SHA256(protocol_digest|beacon|experiment|stream|replicate)"},
    }


def _execution(*, arcs_received_oracle: bool = False, compute_budget_closed: bool = True) -> dict:
    rows = []
    for idx in range(32):
        arcs_u = 1.0e-6 + (idx % 4) * 0.01e-6
        oracle_u = arcs_u * (1.10 + (idx % 3) * 0.01)
        rows.append(
            {
                "replicate": 10_000 + idx,
                "paired_batch_digest": f"pilot-{idx}",
                "world_pairing_closed": True,
                "arcs_branch": {
                    "verified_solution_rate": 0.75,
                    "verified_decision_accuracy": 0.75,
                    "accounted_flops_per_episode": 1000,
                    "verified_utility_per_accounted_flop": arcs_u,
                },
                "oracle_cbrf": {
                    "verified_solution_rate": 0.75,
                    "verified_decision_accuracy": 0.75,
                    "accounted_flops_per_episode": 1000,
                    "verified_utility_per_accounted_flop": oracle_u,
                },
                "oracle_relative_verified_utility_gain": 0.10,
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
        "root_seed": "exp277-dev-seed",
        "model_init_seed": 1234567,
        "initial_state": {
            "arcs_branch_digest": "i" * 64,
            "oracle_cbrf_digest": "i" * 64,
            "functional_digest_match": True,
        },
        "final_state": {"arcs_branch_digest": "h" * 64, "oracle_cbrf_digest": "o" * 64},
        "arm_geometry": {"d_model": 64, "hidden_size": 48, "target_parameters": 500000},
        "world_geometry": {
            "batch_size": 8,
            "timesteps": 4,
            "variables": 6,
            "constraints": 3,
            "d_model": 64,
            "noise_std": 0.05,
        },
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "same_world_lineage": True,
            "compute_budget_closed": compute_budget_closed,
            "declared_max_accounted_flops_per_episode": 1000,
            "pair_audit": {
                "schema": "NLM-EXP-277-MATCHED-ARMS-DEV-V1",
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "parameter_match": True,
                "functional_parameter_match": True,
                "oracle_information_separation": not arcs_received_oracle,
                "compute_budget_closed": compute_budget_closed,
            },
        },
        "oracle_information_receipt": {
            "artifact": "oracle_incidence",
            "ground_truth": True,
            "delivered_to": ["oracle_cbrf"],
            "withheld_from": ["arcs_branch"],
            "arcs_received_oracle_incidence": arcs_received_oracle,
        },
        "training": {
            "rng_stream": "augmentation",
            "start_replicate": 0,
            "replicates": 16,
            "paired_batch_digests": [f"train-{i}" for i in range(16)],
            "optimizer": {"type": "AdamW", "lr": 0.002, "weight_decay": 0.0},
            "arcs_branch_losses": [0.5] * 16,
            "oracle_cbrf_losses": [0.5] * 16,
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": 10_000,
            "replicates": 32,
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
                "paired_execution_evidence": {"artifact_digest": "a" * 64, "code_digest": "c" * 64},
            }
        },
    }


def _prepared(execution: dict | None = None) -> dict:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    execution = execution or _execution()
    return build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=execution,
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )


def _checkpoint_receipt(execution: dict) -> dict:
    from nolane_ai.experiments.exp277_checkpoint import (
        _execution_contract,
        _receipt_digest,
        _scientific_identity_digest,
    )
    from nolane_ai.protocol.evidence import canonical_sha256

    contract = _execution_contract(execution)
    contract_digest = canonical_sha256(contract)
    receipt = {
        "schema": "NLM-EXP-277-TRAINED-CHECKPOINT-V1",
        "experiment_id": "EXP-277",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "state_policy": "functional-only",
        "training_replay_verified": True,
        "execution_contract": contract,
        "execution_contract_digest": contract_digest,
        "arcs_branch_final_digest": execution["final_state"]["arcs_branch_digest"],
        "oracle_cbrf_final_digest": execution["final_state"]["oracle_cbrf_digest"],
        "scientific_identity_digest": "",
        "checkpoint_file_sha256": "f" * 64,
        "receipt_digest": "",
    }
    receipt["scientific_identity_digest"] = _scientific_identity_digest(
        execution_contract_digest=contract_digest,
        arcs_branch_final_digest=receipt["arcs_branch_final_digest"],
        oracle_cbrf_final_digest=receipt["oracle_cbrf_final_digest"],
    )
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def _machinery() -> dict[str, str]:
    return {
        "challenge_generator_digest": "1" * 64,
        "beacon_seed_derivation_digest": "2" * 64,
        "executor_code_digest": "3" * 64,
        "reconstruction_code_digest": "4" * 64,
    }


def _build():
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        build_exp277_gate_a_authorization,
        build_exp277_gate_a_seal,
    )

    execution = _execution()
    prep = _prepared(execution)
    checkpoint = _checkpoint_receipt(execution)
    authorization = build_exp277_gate_a_authorization(
        prep_artifact=prep,
        checkpoint_receipt=checkpoint,
        development_execution_artifact=execution,
        arm_registry=_registry(),
        source_tree_digest="s" * 64,
        freeze_commit_sha="a" * 40,
        freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
        checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
        machinery_digests=_machinery(),
    )
    seal = build_exp277_gate_a_seal(authorization=authorization)
    return execution, prep, checkpoint, authorization, seal


def test_exp277_gate_a_authorization_and_seal_freeze_without_consuming_challenge_data() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        validate_exp277_gate_a_authorization,
        validate_exp277_gate_a_seal,
    )

    _, prep, checkpoint, authorization, seal = _build()
    assert authorization["schema"] == "NLM-EXP-277-CONFIRMATORY-GATE-A-AUTH-V1"
    assert authorization["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["challenge_materialized"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["decision_rule_executed"] is False
    assert authorization["checkpoint"]["receipt_digest"] == checkpoint["receipt_digest"]
    assert authorization["confirmatory_n"] == prep["sample_size_freeze"]["confirmatory_n"]
    assert authorization["reserved_replicate_ids"] == prep["confirmatory_lineage"]["reserved_replicate_ids"]
    assert validate_exp277_gate_a_authorization(authorization) == []

    assert seal["schema"] == "NLM-EXP-277-CONFIRMATORY-GATE-A-SEAL-V1"
    assert seal["status"] == "CONFIRMATORY_GATE_A_SEALED"
    assert seal["readiness"] == "FROZEN_MACHINERY_AND_CHECKPOINT_READY_FOR_FUTURE_BEACON_ONLY"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["decision_rule_executed"] is False
    assert validate_exp277_gate_a_seal(seal) == []
    rendered = repr(seal).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered
    assert "challenge_batch" not in rendered


def test_exp277_gate_a_binds_frozen_analysis_resource_oracle_and_machinery_contracts() -> None:
    from nolane_ai.protocol.evidence import canonical_sha256

    execution, prep, checkpoint, authorization, seal = _build()
    assert authorization["protocol_digest"] == execution["protocol_digest"]
    assert authorization["source_tree_digest"] == "s" * 64
    assert authorization["development_execution_digest"] == execution["artifact_digest"]
    assert authorization["arm_registry_digest"] == _registry()["registry_digest"]
    assert authorization["prep_digest"] == prep["prep_digest"]
    assert authorization["frozen_analysis_digest"] == canonical_sha256(prep["frozen_analysis"])
    assert authorization["checkpoint"]["scientific_identity_digest"] == checkpoint["scientific_identity_digest"]
    assert authorization["resource_court"]["compute_budget_closed"] is True
    assert authorization["resource_court"]["parameter_match"] is True
    assert authorization["oracle_information_separation"]["arcs_received_oracle_incidence"] is False
    assert authorization["endpoint_alpha"] == pytest.approx(0.025)
    assert authorization["bootstrap_samples"] == 10_000
    assert authorization["machinery_digests"] == _machinery()
    assert seal["authorization_digest"] == authorization["authorization_digest"]


def test_exp277_gate_a_rejects_not_ready_prep_and_resource_or_oracle_court_failure() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import build_exp277_gate_a_authorization
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    bad_baseline = _execution()
    for row in bad_baseline["evaluation"]["per_replicate"]:
        row["arcs_branch"]["verified_utility_per_accounted_flop"] = 0.0
    not_ready = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=bad_baseline,
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    with pytest.raises(ValueError, match="not executable"):
        build_exp277_gate_a_authorization(
            prep_artifact=not_ready,
            checkpoint_receipt=_checkpoint_receipt(bad_baseline),
            development_execution_artifact=bad_baseline,
            arm_registry=_registry(),
            source_tree_digest="s" * 64,
            freeze_commit_sha="a" * 40,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
            machinery_digests=_machinery(),
        )

    good_execution = _execution()
    good_prep = _prepared(good_execution)
    court_open = deepcopy(good_execution)
    court_open["resource_match"]["compute_budget_closed"] = False
    with pytest.raises(ValueError, match="resource"):
        build_exp277_gate_a_authorization(
            prep_artifact=good_prep,
            checkpoint_receipt=_checkpoint_receipt(good_execution),
            development_execution_artifact=court_open,
            arm_registry=_registry(),
            source_tree_digest="s" * 64,
            freeze_commit_sha="a" * 40,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
            machinery_digests=_machinery(),
        )

    oracle_open = deepcopy(good_execution)
    oracle_open["oracle_information_receipt"]["arcs_received_oracle_incidence"] = True
    with pytest.raises(ValueError, match="oracle"):
        build_exp277_gate_a_authorization(
            prep_artifact=good_prep,
            checkpoint_receipt=_checkpoint_receipt(good_execution),
            development_execution_artifact=oracle_open,
            arm_registry=_registry(),
            source_tree_digest="s" * 64,
            freeze_commit_sha="a" * 40,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
            machinery_digests=_machinery(),
        )


def test_exp277_gate_a_rejects_checkpoint_or_development_lineage_mismatch() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import build_exp277_gate_a_authorization
    from nolane_ai.experiments.exp277_checkpoint import _receipt_digest

    execution = _execution()
    prep = _prepared(execution)
    checkpoint = _checkpoint_receipt(execution)
    checkpoint["arcs_branch_final_digest"] = "x" * 64
    checkpoint["receipt_digest"] = _receipt_digest(checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        build_exp277_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=checkpoint,
            development_execution_artifact=execution,
            arm_registry=_registry(),
            source_tree_digest="s" * 64,
            freeze_commit_sha="a" * 40,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
            machinery_digests=_machinery(),
        )

    drift = deepcopy(execution)
    drift["artifact_digest"] = "z" * 64
    with pytest.raises(ValueError, match="development"):
        build_exp277_gate_a_authorization(
            prep_artifact=prep,
            checkpoint_receipt=_checkpoint_receipt(execution),
            development_execution_artifact=drift,
            arm_registry=_registry(),
            source_tree_digest="s" * 64,
            freeze_commit_sha="a" * 40,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
            machinery_digests=_machinery(),
        )


def test_exp277_gate_a_validators_reject_rehashed_semantic_tamper_and_prefreeze_material() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        _authorization_digest,
        _seal_digest,
        validate_exp277_gate_a_authorization,
        validate_exp277_gate_a_seal,
    )

    _, _, _, authorization, seal = _build()
    tampered = deepcopy(authorization)
    tampered["endpoint_alpha"] = 0.05
    tampered["authorization_digest"] = _authorization_digest(tampered)
    assert any("alpha" in error.lower() for error in validate_exp277_gate_a_authorization(tampered))

    tampered = deepcopy(authorization)
    tampered["machinery_digests"]["executor_code_digest"] = "9" * 64
    tampered["authorization_digest"] = _authorization_digest(tampered)
    assert any("binding" in error.lower() or "machinery" in error.lower() for error in validate_exp277_gate_a_authorization(tampered))

    bad_seal = deepcopy(seal)
    bad_seal["beacon_receipt"] = {"entropy_hex": "ab" * 32}
    bad_seal["seal_digest"] = _seal_digest(bad_seal)
    assert any("forbidden" in error.lower() or "beacon" in error.lower() for error in validate_exp277_gate_a_seal(bad_seal))


def test_exp277_gate_a_requires_ordered_utc_freeze_and_checkpoint_seal_times() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import build_exp277_gate_a_authorization

    execution = _execution()
    kwargs = dict(
        prep_artifact=_prepared(execution),
        checkpoint_receipt=_checkpoint_receipt(execution),
        development_execution_artifact=execution,
        arm_registry=_registry(),
        source_tree_digest="s" * 64,
        freeze_commit_sha="a" * 40,
        machinery_digests=_machinery(),
    )
    with pytest.raises(ValueError, match="checkpoint seal"):
        build_exp277_gate_a_authorization(
            **kwargs,
            freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
            checkpoint_seal_created_at_utc="2026-09-08T09:59:59Z",
        )
    with pytest.raises(ValueError, match="UTC"):
        build_exp277_gate_a_authorization(
            **kwargs,
            freeze_commit_timestamp_utc="2026-09-08T12:00:00+02:00",
            checkpoint_seal_created_at_utc="2026-09-08T12:00:30+02:00",
        )
