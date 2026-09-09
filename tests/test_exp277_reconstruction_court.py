from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


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
        "failure_policy": {
            "scientific_failure_kept": True,
            "rerun_only": "predeclared infrastructure failure before first model update/inference step",
        },
        "challenge_generator": {
            "lane": "POST_FREEZE_CHALLENGE",
            "seed_rule": "SHA256(protocol_digest|beacon|experiment|stream|replicate)",
        },
    }


def _build_development_and_checkpoint(path: Path):
    from nolane_ai.experiments.exp277_checkpoint import (
        STATE_POLICY,
        TENSOR_SCHEMA,
        _execution_contract,
        _functional_state,
        _receipt_digest,
        _scientific_identity_digest,
    )
    from nolane_ai.experiments.exp277_paired_runner import _build_seeded_pair, _functional_state_digest
    from nolane_ai.experiments.matched_cbrf_arms import audit_matched_exp277_arm_pair
    from nolane_ai.protocol.evidence import canonical_sha256
    from nolane_ai.protocol.identity import file_sha256

    root_seed = "exp277-reconstruction-test-root"
    d_model = 8
    hidden_size = 8
    target_parameters = 100_000
    world = {
        "batch_size": 2,
        "timesteps": 3,
        "variables": 4,
        "constraints": 2,
        "d_model": d_model,
        "noise_std": 0.01,
    }
    arcs, oracle, model_init_seed = _build_seeded_pair(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
    )
    audit = audit_matched_exp277_arm_pair(
        arcs,
        oracle,
        timesteps=world["timesteps"],
        variables=world["variables"],
        constraints=world["constraints"],
    )
    arcs_flops = int(audit["compute_ledger"]["arcs_branch"]["accounted_flops_per_episode"])
    oracle_flops = int(audit["compute_ledger"]["oracle_cbrf"]["accounted_flops_per_episode"])
    ceiling = int(audit["declared_max_accounted_flops_per_episode"])
    arcs_digest = _functional_state_digest(arcs)
    oracle_digest = _functional_state_digest(oracle)

    rows = []
    for idx in range(32):
        arcs_utility = 1.0 / arcs_flops
        oracle_utility = arcs_utility * (1.10 + (idx % 2) * 0.002)
        rows.append(
            {
                "replicate": 10_000 + idx,
                "paired_batch_digest": f"pilot-{idx}",
                "world_pairing_closed": True,
                "arcs_branch": {
                    "verified_solution_rate": 1.0,
                    "verified_decision_accuracy": 1.0,
                    "accounted_flops_per_episode": arcs_flops,
                    "verified_utility_per_accounted_flop": arcs_utility,
                },
                "oracle_cbrf": {
                    "verified_solution_rate": 1.0,
                    "verified_decision_accuracy": 1.0,
                    "accounted_flops_per_episode": oracle_flops,
                    "verified_utility_per_accounted_flop": oracle_utility,
                },
                "oracle_relative_verified_utility_gain": (oracle_utility - arcs_utility) / arcs_utility,
            }
        )

    execution = {
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
        "root_seed": root_seed,
        "model_init_seed": model_init_seed,
        "initial_state": {
            "arcs_branch_digest": arcs_digest,
            "oracle_cbrf_digest": oracle_digest,
            "functional_digest_match": arcs_digest == oracle_digest,
        },
        "final_state": {
            "arcs_branch_digest": arcs_digest,
            "oracle_cbrf_digest": oracle_digest,
        },
        "arm_geometry": {
            "d_model": d_model,
            "hidden_size": hidden_size,
            "target_parameters": target_parameters,
        },
        "world_geometry": world,
        "resource_match": {
            "parameter_match": True,
            "functional_parameter_match": True,
            "same_world_lineage": True,
            "compute_budget_closed": True,
            "declared_max_accounted_flops_per_episode": ceiling,
            "pair_audit": audit,
            "pair_audit_digest": canonical_sha256(audit),
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
            "replicates": 1,
            "paired_batch_digests": ["synthetic-training-lineage"],
            "optimizer": {"type": "AdamW", "lr": 0.002, "weight_decay": 0.0},
            "arcs_branch_losses": [0.5],
            "oracle_cbrf_losses": [0.5],
        },
        "evaluation": {
            "rng_stream": "evaluation",
            "start_replicate": 10_000,
            "replicates": 32,
            "per_replicate": rows,
        },
    }

    contract = _execution_contract(execution)
    contract_digest = canonical_sha256(contract)
    tensor_payload = {
        "schema": TENSOR_SCHEMA,
        "state_policy": STATE_POLICY,
        "execution_contract": contract,
        "execution_contract_digest": contract_digest,
        "arcs_branch_state": _functional_state(arcs),
        "oracle_cbrf_state": _functional_state(oracle),
    }
    torch.save(tensor_payload, path)
    receipt = {
        "schema": "NLM-EXP-277-TRAINED-CHECKPOINT-V1",
        "experiment_id": "EXP-277",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "state_policy": STATE_POLICY,
        "training_replay_verified": True,
        "execution_contract": contract,
        "execution_contract_digest": contract_digest,
        "arcs_branch_final_digest": arcs_digest,
        "oracle_cbrf_final_digest": oracle_digest,
        "scientific_identity_digest": _scientific_identity_digest(
            execution_contract_digest=contract_digest,
            arcs_branch_final_digest=arcs_digest,
            oracle_cbrf_final_digest=oracle_digest,
        ),
        "checkpoint_file_sha256": file_sha256(path),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return execution, audit, receipt, arcs_flops, oracle_flops


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


def _beacon() -> dict:
    from nolane_ai.experiments.exp277_beacon import _receipt_digest

    payload = {
        "schema": "NLM-EXP-277-PUBLIC-BEACON-RECEIPT-V1",
        "experiment_id": "EXP-277",
        "source": "public-beacon.example",
        "beacon_id": "round-277",
        "published_at_utc": "2026-09-08T10:01:00Z",
        "entropy_hex": "ab" * 32,
        "evidence_reference": "https://public-beacon.example/round/277",
        "test_only": False,
        "scientific_evidence_eligible": True,
        "authenticity_status": "EXTERNAL_EVIDENCE_RECORDED",
        "receipt_digest": "",
    }
    payload["receipt_digest"] = _receipt_digest(payload)
    return payload


@pytest.fixture(scope="module")
def court_fixture(tmp_path_factory):
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        build_exp277_gate_a_authorization,
        build_exp277_gate_a_seal,
    )
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    checkpoint_path = tmp_path_factory.mktemp("exp277-reconstruction") / "checkpoint.pt"
    execution, audit, checkpoint_receipt, arcs_flops, oracle_flops = _build_development_and_checkpoint(checkpoint_path)
    prep = build_exp277_confirmatory_prep(
        experiment=_protocol_exp277(),
        execution_artifact=execution,
        arm_registry=_registry(),
        analysis_code_digest="d" * 64,
    )
    authorization = build_exp277_gate_a_authorization(
        prep_artifact=prep,
        checkpoint_receipt=checkpoint_receipt,
        development_execution_artifact=execution,
        arm_registry=_registry(),
        source_tree_digest="s" * 64,
        freeze_commit_sha="a" * 40,
        freeze_commit_timestamp_utc="2026-09-08T10:00:00Z",
        checkpoint_seal_created_at_utc="2026-09-08T10:00:30Z",
        machinery_digests={
            "challenge_generator_digest": "1" * 64,
            "beacon_seed_derivation_digest": "2" * 64,
            "executor_code_digest": "3" * 64,
            "reconstruction_code_digest": "4" * 64,
        },
    )
    seal = build_exp277_gate_a_seal(authorization=authorization)
    return {
        "checkpoint_path": checkpoint_path,
        "execution": execution,
        "audit": audit,
        "checkpoint_receipt": checkpoint_receipt,
        "prep": prep,
        "authorization": authorization,
        "seal": seal,
        "beacon": _beacon(),
        "arcs_flops": arcs_flops,
        "oracle_flops": oracle_flops,
    }


def test_exp277_reconstruction_authorization_binds_exact_per_arm_flops_and_gate_a_lineage(court_fixture) -> None:
    from nolane_ai.experiments.exp277_reconstruction_court import (
        build_exp277_reconstruction_authorization,
        validate_exp277_reconstruction_authorization,
    )

    fixture = court_fixture
    authorization = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    assert authorization["schema"] == "NLM-EXP-277-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
    assert authorization["status"] == "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["challenge_materialized"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["decision_rule_executed"] is False
    assert authorization["seal_digest"] == fixture["seal"]["seal_digest"]
    assert authorization["checkpoint_scientific_identity_digest"] == fixture["checkpoint_receipt"]["scientific_identity_digest"]
    assert authorization["arm_accounted_flops_per_episode"] == {
        "arcs_branch": fixture["arcs_flops"],
        "oracle_cbrf": fixture["oracle_flops"],
    }
    assert validate_exp277_reconstruction_authorization(authorization) == []


def test_exp277_gate_a_snapshot_contains_exact_per_arm_flops_from_pair_audit(court_fixture) -> None:
    fixture = court_fixture
    resource = fixture["authorization"]["resource_court"]
    assert resource["arm_accounted_flops_per_episode"] == {
        "arcs_branch": fixture["arcs_flops"],
        "oracle_cbrf": fixture["oracle_flops"],
    }
    assert resource["arm_accounted_flops_per_episode"]["arcs_branch"] == fixture["audit"]["compute_ledger"]["arcs_branch"]["accounted_flops_per_episode"]
    assert resource["arm_accounted_flops_per_episode"]["oracle_cbrf"] == fixture["audit"]["compute_ledger"]["oracle_cbrf"]["accounted_flops_per_episode"]


def test_exp277_reconstruction_regenerates_one_raw_row_from_seal_beacon_and_checkpoint(court_fixture) -> None:
    from nolane_ai.experiments.exp277_reconstruction_court import (
        build_exp277_reconstruction_authorization,
        reconstruct_exp277_expected_row,
        validate_exp277_raw_row_against_reconstruction,
    )

    fixture = court_fixture
    reconstruction = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    replicate = fixture["seal"]["reserved_replicate_ids"][0]
    row = reconstruct_exp277_expected_row(
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        replicate=replicate,
    )
    assert row["replicate"] == replicate
    assert row["challenge_seed"] >= 0
    assert len(row["challenge_digest"]) == 64
    assert row["checkpoint_scientific_identity_digest"] == fixture["checkpoint_receipt"]["scientific_identity_digest"]
    assert row["oracle_information_receipt"]["arcs_received_oracle_incidence"] is False
    assert row["arcs_branch"]["accounted_flops_per_episode"] == fixture["arcs_flops"]
    assert row["oracle_cbrf"]["accounted_flops_per_episode"] == fixture["oracle_flops"]
    assert row["arcs_branch"]["verified_utility_per_accounted_flop"] == pytest.approx(
        row["arcs_branch"]["verified_solution_rate"] / fixture["arcs_flops"]
    )
    assert row["oracle_cbrf"]["verified_utility_per_accounted_flop"] == pytest.approx(
        row["oracle_cbrf"]["verified_solution_rate"] / fixture["oracle_flops"]
    )
    assert validate_exp277_raw_row_against_reconstruction(
        raw_row=row,
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


@pytest.mark.parametrize(
    ("tamper", "expected_term"),
    [
        ("replicate", "replicate"),
        ("seed", "seed"),
        ("challenge_digest", "challenge"),
        ("predictions", "prediction"),
        ("solution_rate", "solution"),
        ("flops", "flop"),
        ("utility", "utility"),
        ("paired", "paired"),
        ("checkpoint", "checkpoint"),
        ("oracle_leakage", "oracle"),
    ],
)
def test_exp277_reconstruction_rejects_rehashed_forged_raw_rows(court_fixture, tamper: str, expected_term: str) -> None:
    from nolane_ai.experiments.exp277_reconstruction_court import (
        _row_digest,
        build_exp277_reconstruction_authorization,
        reconstruct_exp277_expected_row,
        validate_exp277_raw_row_against_reconstruction,
    )

    fixture = court_fixture
    reconstruction = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    row = reconstruct_exp277_expected_row(
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        replicate=fixture["seal"]["reserved_replicate_ids"][0],
    )
    forged = deepcopy(row)
    if tamper == "replicate":
        forged["replicate"] += 1
    elif tamper == "seed":
        forged["challenge_seed"] += 1
    elif tamper == "challenge_digest":
        forged["challenge_digest"] = "0" * 64
    elif tamper == "predictions":
        forged["arcs_branch"]["predictions"][0][0] = 1 - forged["arcs_branch"]["predictions"][0][0]
    elif tamper == "solution_rate":
        forged["arcs_branch"]["verified_solution_rate"] = 0.123
    elif tamper == "flops":
        forged["oracle_cbrf"]["accounted_flops_per_episode"] += 1
    elif tamper == "utility":
        forged["arcs_branch"]["verified_utility_per_accounted_flop"] += 1e-9
    elif tamper == "paired":
        forged["paired"]["solution_rate_difference"] += 0.1
    elif tamper == "checkpoint":
        forged["checkpoint_scientific_identity_digest"] = "9" * 64
    elif tamper == "oracle_leakage":
        forged["oracle_information_receipt"]["arcs_received_oracle_incidence"] = True
    else:  # pragma: no cover
        raise AssertionError(tamper)
    forged["row_digest"] = _row_digest(forged)

    errors = validate_exp277_raw_row_against_reconstruction(
        raw_row=forged,
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert errors
    assert any(expected_term in error.lower() for error in errors)


def test_exp277_reconstruction_rejects_checkpoint_or_beacon_lineage_mismatch(court_fixture) -> None:
    from nolane_ai.experiments.exp277_beacon import _receipt_digest as beacon_digest
    from nolane_ai.experiments.exp277_reconstruction_court import (
        build_exp277_reconstruction_authorization,
        reconstruct_exp277_expected_row,
    )

    fixture = court_fixture
    reconstruction = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )
    bad_checkpoint = deepcopy(fixture["checkpoint_receipt"])
    bad_checkpoint["scientific_identity_digest"] = "8" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        reconstruct_exp277_expected_row(
            reconstruction_authorization=reconstruction,
            seal=fixture["seal"],
            beacon_receipt=fixture["beacon"],
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=bad_checkpoint,
            replicate=fixture["seal"]["reserved_replicate_ids"][0],
        )

    bad_beacon = deepcopy(fixture["beacon"])
    bad_beacon["published_at_utc"] = "2026-09-08T10:00:30Z"
    bad_beacon["receipt_digest"] = beacon_digest(bad_beacon)
    with pytest.raises(ValueError, match="beacon"):
        reconstruct_exp277_expected_row(
            reconstruction_authorization=reconstruction,
            seal=fixture["seal"],
            beacon_receipt=bad_beacon,
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            replicate=fixture["seal"]["reserved_replicate_ids"][0],
        )
