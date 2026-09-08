from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    authorize_exp297_confirmatory_execution,
    validate_exp297_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp297_confirmatory_prep import build_exp297_confirmatory_prep
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_reconstruction_court import (
    authorize_exp297_confirmatory_reconstruction,
    validate_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prep():
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    experiment = next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-297")
    execution = run_exp297_paired_development(
        root_seed="exp297-auth-test",
        eval_replicates=32,
        eval_start_replicate=6000,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )
    registry = {
        "registry_digest": canonical_sha256({"execution": execution["artifact_digest"]}),
        "experiments": {
            "EXP-297": {
                "evidence_level": "EV-E2",
                "decision": "UNVERIFIED",
                "match_court": "BLOCKED",
                "paired_execution_evidence": {"execution_digest": execution["artifact_digest"]},
            }
        },
    }
    return build_exp297_confirmatory_prep(
        experiment=experiment,
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest="a" * 64,
    )


def _auth(prep):
    return authorize_exp297_confirmatory_execution(
        prep_artifact=prep,
        challenge_contract_digest=challenge_contract_digest(),
        evaluator_code_digest="e" * 64,
        execution_code_digest="x" * 64,
    )


def _rehash_auth(payload):
    clean = deepcopy(payload)
    clean.pop("authorization_digest", None)
    payload["authorization_digest"] = canonical_sha256(clean)


def _rehash_reconstruction(payload):
    clean = deepcopy(payload)
    clean.pop("reconstruction_digest", None)
    payload["reconstruction_digest"] = canonical_sha256(clean)


def test_exp297_execution_authorization_freezes_geometry_without_materializing_beacon_or_seeds(prep):
    authorization = _auth(prep)
    assert authorization["schema"] == "NLM-EXP-297-CONFIRMATORY-EXECUTION-AUTH-V1"
    assert authorization["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["challenge_materialized"] is False
    assert authorization["decision_rule_executed"] is False
    assert authorization["semantic_authority_promoted"] is False
    assert authorization["candidate_count_per_replicate"] == 16
    assert authorization["challenge_contract_digest"] == challenge_contract_digest()
    assert authorization["model_geometry"] == {
        "d_model": 8,
        "hidden_size": 8,
        "target_parameters": 8_000,
        "max_exact_assignments": 4096,
    }
    rendered = repr(authorization).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered
    assert "challenge_candidates" not in rendered
    assert validate_exp297_confirmatory_execution_authorization(authorization) == []


def test_exp297_execution_authorization_rejects_rehashed_lineage_geometry_and_materialization_tampering(prep):
    authorization = _auth(prep)
    mutations = []
    changed = deepcopy(authorization)
    changed["reserved_replicate_ids"][0] += 1
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["challenge_contract_digest"] = "0" * 64
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["model_geometry"]["d_model"] = 16
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["candidate_count_per_replicate"] = 15
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["lineage"]["pair_audit_digest"] = "0" * 64
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["seed_materialization_status"] = "EXECUTED"
    mutations.append(changed)
    changed = deepcopy(authorization)
    changed["challenge_materialized"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash_auth(changed)
        assert validate_exp297_confirmatory_execution_authorization(changed)


def test_exp297_reconstruction_authorization_binds_execution_authority_and_exact_rebuild_contract(prep):
    authorization = _auth(prep)
    reconstruction = authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest="e" * 64,
    )
    assert reconstruction["schema"] == "NLM-EXP-297-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
    assert reconstruction["status"] == "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
    assert reconstruction["confirmatory_data_consumed"] is False
    assert reconstruction["seed_materialization_status"] == "NOT_EXECUTED"
    assert reconstruction["challenge_materialized"] is False
    contract = reconstruction["reconstruction_contract"]
    assert contract["candidate_count_per_replicate"] == 16
    assert contract["reconstruct_beacon_derived_seed"] is True
    assert contract["reconstruct_challenge_world"] is True
    assert contract["reconstruct_exact_fidelity_receipt"] is True
    assert contract["reconstruct_matched_neural_pair"] is True
    assert contract["reconstruct_neural_and_semantic_costs"] is True
    assert contract["reconstruct_primary_and_safety_counts"] is True
    assert validate_exp297_confirmatory_reconstruction(reconstruction) == []


def test_exp297_reconstruction_authorization_rejects_rehashed_authority_and_contract_tampering(prep):
    authorization = _auth(prep)
    reconstruction = authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest="e" * 64,
    )
    mutations = []
    changed = deepcopy(reconstruction)
    changed["lineage"]["execution_authorization_digest"] = "0" * 64
    mutations.append(changed)
    changed = deepcopy(reconstruction)
    changed["reconstruction_contract"]["reconstruct_exact_fidelity_receipt"] = False
    mutations.append(changed)
    changed = deepcopy(reconstruction)
    changed["model_geometry"]["max_exact_assignments"] = 8
    mutations.append(changed)
    changed = deepcopy(reconstruction)
    changed["semantic_authority_promoted"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash_reconstruction(changed)
        assert validate_exp297_confirmatory_reconstruction(changed)
