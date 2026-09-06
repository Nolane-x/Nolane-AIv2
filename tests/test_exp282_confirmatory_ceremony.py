from copy import deepcopy

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
from nolane_ai.experiments.exp282_reconstruction_court import authorize_exp282_confirmatory_reconstruction
from nolane_ai.experiments.exp282_confirmatory_ceremony import (
    _result_digest,
    _seal_digest,
    execute_exp282_confirmatory_ceremony,
    seal_exp282_confirmatory_ceremony,
    validate_exp282_confirmatory_ceremony_result,
    validate_exp282_confirmatory_ceremony_seal,
)
from tests.exp282_confirmatory_fixtures import CANONICAL_PROTOCOL_DIGEST, prepared_chain

CODE_DIGEST = "a" * 64


def _chain(tmp_path):
    protocol, prep, execution, _, _ = prepared_chain(tmp_path, analysis_code_digest=CODE_DIGEST)
    authorization = authorize_exp282_confirmatory_execution(
        prep_artifact=prep,
        execution_code_digest=CODE_DIGEST,
    )
    reconstruction = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=authorization,
    )
    return protocol, prep, execution, authorization, reconstruction


def _seal(tmp_path):
    _, prep, execution, authorization, reconstruction = _chain(tmp_path)
    return seal_exp282_confirmatory_ceremony(
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        paired_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_authorization=reconstruction,
        ceremony_code_digest=CODE_DIGEST,
    )


def test_seal_binds_lineage_without_consuming_confirmatory_data(tmp_path):
    seal = _seal(tmp_path)
    assert seal["schema"] == "NLM-EXP-282-CONFIRMATORY-CEREMONY-SEAL-V1"
    assert seal["evidence_level"] == "EV-E2"
    assert seal["decision"] == "UNVERIFIED"
    assert seal["status"] == "CEREMONY_SEALED_NOT_EXECUTED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["seed_materialization_status"] == "NOT_EXECUTED"
    assert seal["challenge_materialized"] is False
    assert seal["confirmatory_n"] == 32
    assert len(seal["reserved_replicate_ids"]) == 32
    assert seal["lineage"]["protocol_digest"] == CANONICAL_PROTOCOL_DIGEST
    assert seal["lineage"]["ceremony_code_digest"] == CODE_DIGEST
    assert seal["lineage"]["analysis_code_digest"] == CODE_DIGEST
    assert seal["lineage"]["execution_code_digest"] == CODE_DIGEST
    assert validate_exp282_confirmatory_ceremony_seal(seal) == []


def test_seal_rejects_source_tree_digest_drift(tmp_path):
    _, prep, execution, authorization, reconstruction = _chain(tmp_path)
    with pytest.raises(ValueError, match="ceremony code digest"):
        seal_exp282_confirmatory_ceremony(
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            paired_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization=authorization,
            reconstruction_authorization=reconstruction,
            ceremony_code_digest="b" * 64,
        )


def test_seal_rejects_not_ready_prep(tmp_path):
    _, prep, execution, _, _ = _chain(tmp_path)
    prep = deepcopy(prep)
    prep["status"] = "NOT_READY_VARIANCE_EXCEEDS_MAX_N"
    prep["sample_size_freeze"]["confirmatory_n"] = None
    prep["confirmatory_lineage"]["reserved_replicate_ids"] = []
    from nolane_ai.experiments.exp282_confirmatory_prep import _prep_digest

    prep["prep_digest"] = _prep_digest(prep)
    with pytest.raises(ValueError):
        seal_exp282_confirmatory_ceremony(
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            paired_execution_artifact=execution,
            prep_artifact=prep,
            execution_authorization={},
            reconstruction_authorization={},
            ceremony_code_digest=CODE_DIGEST,
        )


def test_seal_tamper_and_rehash_still_fails_semantic_validation(tmp_path):
    seal = _seal(tmp_path)
    seal["reserved_replicate_ids"] = seal["reserved_replicate_ids"][1:]
    seal["ceremony_seal_digest"] = _seal_digest(seal)
    errors = validate_exp282_confirmatory_ceremony_seal(seal)
    assert "ceremony reserved replicate count mismatch" in errors


def test_execute_rejects_source_tree_drift_before_checkpoint_access(tmp_path):
    protocol, _, execution, _, _ = _chain(tmp_path)
    seal = _seal(tmp_path)
    with pytest.raises(ValueError, match="current ceremony code digest"):
        execute_exp282_confirmatory_ceremony(
            protocol=protocol,
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            ceremony_seal=seal,
            paired_execution_artifact=execution,
            checkpoint_path=tmp_path / "does-not-exist.pt",
            ceremony_code_digest="b" * 64,
        )


def test_execute_rejects_checkpoint_sha_before_checkpoint_load(tmp_path):
    protocol, _, execution, _, _ = _chain(tmp_path)
    seal = _seal(tmp_path)
    bad_checkpoint = tmp_path / "bad.pt"
    bad_checkpoint.write_bytes(b"not the sealed checkpoint")
    with pytest.raises(ValueError, match="ceremony checkpoint SHA mismatch"):
        execute_exp282_confirmatory_ceremony(
            protocol=protocol,
            protocol_digest=CANONICAL_PROTOCOL_DIGEST,
            ceremony_seal=seal,
            paired_execution_artifact=execution,
            checkpoint_path=bad_checkpoint,
            ceremony_code_digest=CODE_DIGEST,
        )


def test_execute_returns_append_only_raw_and_analysis_bundle(tmp_path):
    protocol, _, execution, _, _ = _chain(tmp_path)
    seal = _seal(tmp_path)
    result = execute_exp282_confirmatory_ceremony(
        protocol=protocol,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        ceremony_seal=seal,
        paired_execution_artifact=execution,
        checkpoint_path=tmp_path / "paired.pt",
        ceremony_code_digest=CODE_DIGEST,
    )
    assert result["schema"] == "NLM-EXP-282-CONFIRMATORY-CEREMONY-RESULT-V1"
    assert result["status"] == "CEREMONY_EXECUTED_AND_ANALYZED"
    assert result["confirmatory_data_consumed"] is True
    assert result["challenge_materialized"] is False
    assert result["artifacts"]["seal"]["ceremony_seal_digest"] == seal["ceremony_seal_digest"]
    assert result["artifacts"]["raw"]["evidence_level"] == "EV-E2"
    assert result["artifacts"]["raw"]["decision"] == "UNVERIFIED"
    assert result["artifacts"]["analysis"]["evidence_level"] == "EV-E3"
    assert result["decision"] == result["artifacts"]["analysis"]["decision"]
    assert result["lineage"]["raw_artifact_digest"] == result["artifacts"]["raw"]["artifact_digest"]
    assert result["lineage"]["analysis_digest"] == result["artifacts"]["analysis"]["analysis_digest"]
    assert validate_exp282_confirmatory_ceremony_result(result) == []


def test_result_tamper_and_rehash_still_fails_embedded_artifact_validation(tmp_path):
    protocol, _, execution, _, _ = _chain(tmp_path)
    result = execute_exp282_confirmatory_ceremony(
        protocol=protocol,
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        ceremony_seal=_seal(tmp_path),
        paired_execution_artifact=execution,
        checkpoint_path=tmp_path / "paired.pt",
        ceremony_code_digest=CODE_DIGEST,
    )
    result["artifacts"]["raw"]["challenge_materialized"] = True
    result["ceremony_result_digest"] = _result_digest(result)
    errors = validate_exp282_confirmatory_ceremony_result(result)
    assert any("raw artifact" in error or "challenge" in error for error in errors)
