from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_ceremony import (
    seal_exp297_confirmatory_gate_a,
    validate_exp297_confirmatory_gate_a_seal,
)
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    authorize_exp297_confirmatory_execution,
)
from nolane_ai.experiments.exp297_confirmatory_prep import build_exp297_confirmatory_prep
from nolane_ai.experiments.exp297_paired_runner import run_exp297_paired_development
from nolane_ai.experiments.exp297_reconstruction_court import (
    authorize_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CODE_TREE_DIGEST = "d" * 64
FREEZE_SHA = "f" * 40
FREEZE_TIME = "2026-09-08T06:30:00Z"


@pytest.fixture(scope="module")
def authorities():
    protocol = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    protocol_digest = (ROOT / "protocols" / "stage_a_v1.sha256").read_text(encoding="utf-8").strip()
    experiment = next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-297")
    execution = run_exp297_paired_development(
        root_seed="exp297-ceremony-test",
        eval_replicates=32,
        eval_start_replicate=12000,
        d_model=8,
        hidden_size=8,
        target_parameters=8_000,
        max_exact_assignments=4096,
        protocol_digest=protocol_digest,
        code_digest=CODE_TREE_DIGEST,
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
    prep = build_exp297_confirmatory_prep(
        experiment=experiment,
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=CODE_TREE_DIGEST,
    )
    authorization = authorize_exp297_confirmatory_execution(
        prep_artifact=prep,
        challenge_contract_digest=challenge_contract_digest(),
        evaluator_code_digest=CODE_TREE_DIGEST,
        execution_code_digest=CODE_TREE_DIGEST,
    )
    reconstruction = authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_code_digest=CODE_TREE_DIGEST,
    )
    return protocol_digest, execution, prep, authorization, reconstruction


def _seal(authorities):
    protocol_digest, execution, prep, authorization, reconstruction = authorities
    return seal_exp297_confirmatory_gate_a(
        protocol_digest=protocol_digest,
        development_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_authorization=reconstruction,
        code_tree_digest=CODE_TREE_DIGEST,
        freeze_commit_sha=FREEZE_SHA,
        freeze_commit_timestamp_utc=FREEZE_TIME,
    )


def _rehash(payload):
    clean = deepcopy(payload)
    clean.pop("seal_digest", None)
    payload["seal_digest"] = canonical_sha256(clean)


def test_exp297_gate_a_seal_closes_pre_beacon_lineage_without_consumption(authorities):
    seal = _seal(authorities)
    assert seal["schema"] == "NLM-EXP-297-CONFIRMATORY-GATE-A-SEAL-V1"
    assert seal["experiment_id"] == "EXP-297"
    assert seal["status"] == "CONFIRMATORY_GATE_A_SEALED"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_ready"] is True
    assert seal["confirmatory_data_consumed"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["challenge_materialized"] is False
    assert seal["decision_rule_executed"] is False
    assert seal["semantic_authority_promoted"] is False
    assert seal["freeze_commit_sha"] == FREEZE_SHA
    assert seal["freeze_commit_timestamp_utc"] == FREEZE_TIME
    assert seal["code_tree_digest"] == CODE_TREE_DIGEST
    assert seal["challenge_contract_digest"] == challenge_contract_digest()
    assert seal["confirmatory_n"] == len(seal["reserved_replicate_ids"])
    assert seal["lineage"]["protocol_digest"] == authorities[0]
    assert seal["lineage"]["development_execution_digest"] == authorities[1]["artifact_digest"]
    assert seal["lineage"]["prep_digest"] == authorities[2]["prep_digest"]
    assert seal["lineage"]["execution_authorization_digest"] == authorities[3]["authorization_digest"]
    assert seal["lineage"]["reconstruction_digest"] == authorities[4]["reconstruction_digest"]
    assert validate_exp297_confirmatory_gate_a_seal(seal) == []

    rendered = repr(seal).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered
    assert "challenge_candidates" not in rendered


def test_exp297_gate_a_seal_rejects_mixed_code_tree_and_invalid_freeze_identity(authorities):
    protocol_digest, execution, prep, authorization, reconstruction = authorities
    with pytest.raises(ValueError, match="code-tree"):
        seal_exp297_confirmatory_gate_a(
            protocol_digest=protocol_digest,
            development_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization=authorization,
            reconstruction_authorization=reconstruction,
            code_tree_digest="e" * 64,
            freeze_commit_sha=FREEZE_SHA,
            freeze_commit_timestamp_utc=FREEZE_TIME,
        )
    with pytest.raises(ValueError, match="freeze commit SHA"):
        seal_exp297_confirmatory_gate_a(
            protocol_digest=protocol_digest,
            development_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization=authorization,
            reconstruction_authorization=reconstruction,
            code_tree_digest=CODE_TREE_DIGEST,
            freeze_commit_sha="not-a-sha",
            freeze_commit_timestamp_utc=FREEZE_TIME,
        )
    with pytest.raises(ValueError, match="freeze commit timestamp"):
        seal_exp297_confirmatory_gate_a(
            protocol_digest=protocol_digest,
            development_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization=authorization,
            reconstruction_authorization=reconstruction,
            code_tree_digest=CODE_TREE_DIGEST,
            freeze_commit_sha=FREEZE_SHA,
            freeze_commit_timestamp_utc="not-a-timestamp",
        )


def test_exp297_gate_a_seal_rejects_rehashed_lineage_contract_and_flag_tampering(authorities):
    seal = _seal(authorities)
    mutations = []

    changed = deepcopy(seal)
    changed["freeze_commit_sha"] = "a" * 40
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["freeze_commit_timestamp_utc"] = "2026-09-08T06:31:00Z"
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["code_tree_digest"] = "e" * 64
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["challenge_contract_digest"] = "0" * 64
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["lineage"]["frozen_analysis_digest"] = "0" * 64
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["lineage"]["sample_size_freeze_digest"] = "0" * 64
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["reserved_replicate_ids"][0] += 1
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["confirmatory_data_consumed"] = True
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["challenge_materialized"] = True
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["decision_rule_executed"] = True
    mutations.append(changed)

    changed = deepcopy(seal)
    changed["semantic_authority_promoted"] = True
    mutations.append(changed)

    for changed in mutations:
        _rehash(changed)
        assert validate_exp297_confirmatory_gate_a_seal(changed)
