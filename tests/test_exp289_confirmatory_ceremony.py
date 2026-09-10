from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.evidence import canonical_sha256
from tests.exp289_confirmatory_fixtures import (
    CODE_DIGEST,
    FREEZE_SHA,
    FREEZE_TIME,
    TEST_GEOMETRY_DIGEST,
    task5_inputs,
)
from tests.test_exp289_confirmatory_authorization import _authorize


def _api():
    try:
        from nolane_ai.experiments.exp289_confirmatory_ceremony import (
            seal_exp289_confirmatory_gate_a,
            validate_exp289_confirmatory_gate_a_seal,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 immutable pre-beacon Gate-A seal is missing")
    return seal_exp289_confirmatory_gate_a, validate_exp289_confirmatory_gate_a_seal


def _seal(inputs: dict, authorization: dict | None = None, **overrides):
    seal, _ = _api()
    authorization = _authorize(inputs) if authorization is None else authorization
    kwargs = {
        "protocol_digest": inputs["protocol_digest"],
        "development_execution_artifact": inputs["authoritative_execution"],
        "prep_artifact": inputs["prep"],
        "checkpoint_receipt": inputs["checkpoint_receipt"],
        "execution_authorization": authorization,
        "expected_geometry_digest": inputs["expected_geometry_digest"],
        "code_tree_digest": inputs["execution_code_digest"],
        "freeze_commit_sha": FREEZE_SHA,
        "freeze_commit_timestamp_utc": FREEZE_TIME,
    }
    kwargs.update(overrides)
    return seal(**kwargs)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_exp289_gate_a_seal_transitively_binds_checkpoint_and_pre_beacon_authority(task5_inputs: dict) -> None:
    _, validate = _api()
    authorization = _authorize(task5_inputs)
    receipt = task5_inputs["checkpoint_receipt"]
    seal = _seal(task5_inputs, authorization)

    assert seal["schema"] == "NLM-EXP-289-CONFIRMATORY-GATE-A-SEAL-V1"
    assert seal["status"] == "CONFIRMATORY_GATE_A_SEALED"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["confirmatory_ready"] is True
    assert seal["confirmatory_ready_scope"] == "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["challenge_materialized"] is False
    assert seal["decision_rule_executed"] is False
    assert seal["freeze_commit_sha"] == FREEZE_SHA
    assert seal["freeze_commit_timestamp_utc"] == FREEZE_TIME
    assert _parse_utc(seal["seal_created_at_utc"]) > _parse_utc(FREEZE_TIME)
    assert seal["code_tree_digest"] == CODE_DIGEST
    assert seal["confirmatory_n"] == authorization["confirmatory_n"]

    lineage = seal["lineage"]
    assert lineage["protocol_digest"] == task5_inputs["protocol_digest"]
    assert lineage["development_geometry_digest"] == TEST_GEOMETRY_DIGEST
    assert lineage["development_execution_digest"] == task5_inputs["authoritative_execution"]["artifact_digest"]
    assert lineage["prep_digest"] == task5_inputs["prep"]["prep_digest"]
    assert lineage["execution_authorization_digest"] == authorization["authorization_digest"]
    assert lineage["challenge_contract_digest"] == authorization["lineage"]["challenge_contract_digest"]
    assert lineage["checkpoint_receipt_digest"] == receipt["receipt_digest"]
    assert lineage["checkpoint_scientific_identity_digest"] == receipt["scientific_identity_digest"]
    assert lineage["checkpoint_file_sha256"] == receipt["checkpoint_file_sha256"]
    assert lineage["checkpoint_execution_contract_digest"] == receipt["execution_contract_digest"]

    rendered = repr(seal).lower()
    for forbidden in (
        "beacon_receipt",
        "entropy_hex",
        "challenge_seed",
        "challenge_candidates",
        "confirmatory_observations",
    ):
        assert forbidden not in rendered
    assert validate(seal) == []


def test_exp289_gate_a_seal_rejects_code_tree_and_checkpoint_lineage_drift(task5_inputs: dict) -> None:
    with pytest.raises(ValueError, match="code"):
        _seal(task5_inputs, code_tree_digest="b" * 64)

    authorization = _authorize(task5_inputs)
    changed = deepcopy(authorization)
    changed["lineage"]["checkpoint_scientific_identity_digest"] = "0" * 64
    clean = deepcopy(changed)
    clean.pop("authorization_digest", None)
    changed["authorization_digest"] = canonical_sha256(clean)
    with pytest.raises(ValueError, match="checkpoint|authorization|lineage"):
        _seal(task5_inputs, authorization=changed)


def test_exp289_gate_a_seal_validator_rejects_freeze_and_lineage_tampering(task5_inputs: dict) -> None:
    _, validate = _api()
    seal = _seal(task5_inputs)

    changed = deepcopy(seal)
    changed["freeze_commit_sha"] = "not-a-sha"
    assert any("freeze commit sha" in item.lower() for item in validate(changed))

    changed = deepcopy(seal)
    changed["seal_created_at_utc"] = "not-a-timestamp"
    assert any("seal" in item.lower() and "timestamp" in item.lower() for item in validate(changed))

    changed = deepcopy(seal)
    changed["lineage"]["development_geometry_digest"] = "0" * 64
    assert any("geometry" in item.lower() or "digest" in item.lower() for item in validate(changed))

    changed = deepcopy(seal)
    changed["lineage"]["checkpoint_receipt_digest"] = "0" * 64
    assert any("checkpoint" in item.lower() or "digest" in item.lower() for item in validate(changed))


def test_exp289_gate_a_seal_binding_changes_with_freeze_identity(task5_inputs: dict) -> None:
    first = _seal(task5_inputs, freeze_commit_sha="1" * 40)
    second = _seal(task5_inputs, freeze_commit_sha="2" * 40)
    assert first["pre_beacon_binding_digest"] != second["pre_beacon_binding_digest"]
    assert first["seal_digest"] != second["seal_digest"]
