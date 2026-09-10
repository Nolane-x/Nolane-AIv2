from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256, require_frozen_stage_a_v1_sha256
from .exp289_challenge_worlds import challenge_contract_digest
from .exp289_checkpoint import load_exp289_trained_checkpoint, validate_exp289_checkpoint_receipt
from .exp289_confirmatory_prep import validate_exp289_confirmatory_prep
from .exp289_development_geometry import AUTHORITY_SCOPE as GEOMETRY_SCOPE, SCHEMA as GEOMETRY_SCHEMA
from .exp289_paired_runner import _artifact_digest, validate_exp289_paired_development


SCHEMA = "NLM-EXP-289-CONFIRMATORY-EXECUTION-AUTHORIZATION-V1"
EXPERIMENT_ID = "EXP-289"
STATUS = "AUTHORIZED_NOT_EXECUTED"
MIN_N = 32
MAX_N = 128


def _authorization_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def _is_hex_digest(value: Any, *, lengths: set[int] = {64}) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _forbidden_material_present(payload: dict[str, Any]) -> list[str]:
    rendered = repr(payload).lower()
    forbidden = ("beacon_receipt", "entropy_hex", "challenge_seed", "confirmatory_observations")
    return [item for item in forbidden if item in rendered]


def _scientific_execution(
    execution_artifact: dict[str, Any],
    *,
    expected_geometry_digest: str,
    expected_geometry_configuration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    if not _is_hex_digest(expected_geometry_digest):
        raise ValueError("EXP-289 expected authoritative geometry digest must be a 64-hex SHA-256 digest")
    if not isinstance(expected_geometry_configuration, dict) or not expected_geometry_configuration:
        raise ValueError("EXP-289 expected authoritative geometry configuration is required")

    authority = execution_artifact.get("development_geometry_authority")
    if not isinstance(authority, dict):
        raise ValueError("EXP-289 authorization requires authoritative DEVELOPMENT geometry")
    if (
        authority.get("schema") != GEOMETRY_SCHEMA
        or authority.get("authority_scope") != GEOMETRY_SCOPE
        or authority.get("manifest_digest") != expected_geometry_digest
        or authority.get("confirmatory_authority") is not False
    ):
        raise ValueError("EXP-289 authorization authoritative DEVELOPMENT geometry mismatch")

    outer_digest = execution_artifact.get("artifact_digest")
    if not _is_hex_digest(outer_digest) or outer_digest != _artifact_digest(execution_artifact):
        raise ValueError("EXP-289 authoritative DEVELOPMENT execution self-hash mismatch")

    scientific = deepcopy(execution_artifact)
    scientific.pop("development_geometry_authority", None)
    scientific["artifact_digest"] = _artifact_digest(scientific)
    scientific_digest = scientific["artifact_digest"]
    if authority.get("scientific_execution_digest") != scientific_digest:
        raise ValueError("EXP-289 authoritative geometry scientific execution digest mismatch")
    if scientific.get("configuration") != expected_geometry_configuration:
        raise ValueError("EXP-289 authoritative DEVELOPMENT geometry configuration mismatch")

    errors = validate_exp289_paired_development(scientific)
    if errors:
        raise ValueError("invalid EXP-289 authoritative DEVELOPMENT scientific execution: " + "; ".join(errors))
    if scientific.get("evidence_level") != "EV-E2" or scientific.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-289 authorization requires EV-E2 / UNVERIFIED DEVELOPMENT evidence")
    for field in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_seed_materialized",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if scientific.get(field) is not False:
            raise ValueError(f"EXP-289 pre-beacon DEVELOPMENT boundary requires {field}=false")
    return scientific, deepcopy(authority), str(outer_digest), str(scientific_digest)


def _checkpoint_errors(
    *,
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    scientific_execution: dict[str, Any],
    geometry_authority: dict[str, Any],
    outer_execution_digest: str,
    expected_geometry_digest: str,
    expected_geometry_configuration: dict[str, Any],
    execution_code_digest: str,
) -> list[str]:
    errors = validate_exp289_checkpoint_receipt(checkpoint_receipt)
    if errors:
        return ["EXP-289 checkpoint receipt invalid: " + "; ".join(errors)]

    path = Path(checkpoint_path)
    if not path.is_file():
        return ["EXP-289 checkpoint file is missing"]
    if file_sha256(path) != checkpoint_receipt.get("checkpoint_file_sha256"):
        return ["EXP-289 checkpoint file SHA256/hash mismatch"]

    contract = checkpoint_receipt.get("execution_contract") or {}
    contract_authority = contract.get("development_geometry_authority") or {}
    expected = {
        "protocol_digest": scientific_execution.get("protocol_digest"),
        "development_code_digest": execution_code_digest,
        "scientific_execution_digest": scientific_execution.get("artifact_digest"),
        "authoritative_execution_digest": outer_execution_digest,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            errors.append(f"EXP-289 checkpoint execution contract lineage mismatch: {key}")
    if contract.get("configuration") != expected_geometry_configuration:
        errors.append("EXP-289 checkpoint execution contract geometry configuration mismatch")
    if (
        contract_authority.get("schema") != GEOMETRY_SCHEMA
        or contract_authority.get("authority_scope") != GEOMETRY_SCOPE
        or contract_authority.get("manifest_digest") != expected_geometry_digest
        or contract_authority.get("scientific_execution_digest") != geometry_authority.get("scientific_execution_digest")
        or contract_authority.get("confirmatory_authority") is not False
    ):
        errors.append("EXP-289 checkpoint execution contract authoritative geometry mismatch")

    if not errors:
        try:
            load_exp289_trained_checkpoint(checkpoint_path=path, receipt=checkpoint_receipt)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            errors.append(f"EXP-289 checkpoint functional-state verification failed: {exc}")
    return errors


def validate_exp289_confirmatory_execution_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-289 execution authorization must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-289 confirmatory execution authorization identity")
    if payload.get("status") != STATUS:
        errors.append("EXP-289 authorization must remain pre-execution")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-289 authorization cannot promote scientific evidence")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-289 authorization cannot consume confirmatory data")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-289 authorization cannot materialize challenge seed entropy")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-289 authorization cannot materialize challenge observations")
    for item in _forbidden_material_present(payload):
        errors.append(f"EXP-289 authorization contains forbidden pre-beacon material: {item}")

    confirmatory_n = payload.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not MIN_N <= confirmatory_n <= MAX_N:
        errors.append("EXP-289 authorization confirmatory_n is outside frozen bounds")

    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        errors.append("EXP-289 authorization lineage missing")
        return errors

    digest_fields = (
        "protocol_digest",
        "development_geometry_digest",
        "development_geometry_configuration_digest",
        "development_execution_digest",
        "scientific_execution_digest",
        "prep_digest",
        "execution_code_digest",
        "challenge_contract_digest",
        "checkpoint_receipt_digest",
        "checkpoint_scientific_identity_digest",
        "checkpoint_file_sha256",
        "checkpoint_execution_contract_digest",
        "checkpoint_no_nogood_final_digest",
        "checkpoint_local_nogood_final_digest",
    )
    for key in digest_fields:
        if not _is_hex_digest(lineage.get(key)):
            errors.append(f"EXP-289 authorization lineage invalid/missing digest: {key}")
    try:
        require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))
    except ValueError as exc:
        errors.append(str(exc))
    if lineage.get("challenge_contract_digest") != challenge_contract_digest():
        errors.append("EXP-289 authorization challenge-contract digest mismatch")

    preflight = payload.get("preflight") or {}
    expected_checks = (
        "development_artifact_valid",
        "prep_artifact_valid",
        "prep_ready",
        "authoritative_geometry_bound",
        "geometry_configuration_bound",
        "development_prep_lineage_bound",
        "protocol_lineage_bound",
        "code_lineage_bound",
        "challenge_contract_bound",
        "checkpoint_receipt_valid",
        "checkpoint_file_hash_bound",
        "checkpoint_execution_contract_bound",
        "checkpoint_functional_state_verified",
        "pre_beacon_boundary_preserved",
        "all_checks_passed",
    )
    for key in expected_checks:
        if preflight.get(key) is not True:
            errors.append(f"EXP-289 authorization preflight failed: {key}")
    if payload.get("authorization_digest") != _authorization_digest(payload):
        errors.append("EXP-289 authorization digest mismatch")
    return errors


def authorize_exp289_confirmatory_execution(
    *,
    execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    expected_geometry_digest: str,
    expected_geometry_configuration: dict[str, Any],
    execution_code_digest: str,
) -> dict[str, Any]:
    if not _is_hex_digest(execution_code_digest):
        raise ValueError("EXP-289 execution code digest must be a 64-hex SHA-256 digest")

    scientific, authority, outer_digest, scientific_digest = _scientific_execution(
        execution_artifact,
        expected_geometry_digest=expected_geometry_digest,
        expected_geometry_configuration=expected_geometry_configuration,
    )
    protocol_digest = scientific.get("protocol_digest")
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if scientific.get("code_digest") != execution_code_digest:
        raise ValueError("EXP-289 authorization execution code lineage mismatch")

    prep_errors = validate_exp289_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-289 confirmatory prep: " + "; ".join(prep_errors))
    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep_artifact.get("confirmatory_ready") is not True:
        raise ValueError("EXP-289 authorization requires ready Gate-A prep")
    if prep_artifact.get("evidence_level") != "EV-E2" or prep_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-289 authorization requires EV-E2 / UNVERIFIED Gate-A prep")
    if prep_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 authorization protocol/prep lineage mismatch")
    if prep_artifact.get("analysis_code_digest") != execution_code_digest:
        raise ValueError("EXP-289 authorization prep/code lineage mismatch")
    if prep_artifact.get("development_execution_digest") != scientific_digest:
        raise ValueError("EXP-289 authorization scientific execution/prep lineage mismatch")

    prep_authority = prep_artifact.get("development_geometry_authority")
    if not isinstance(prep_authority, dict):
        raise ValueError("EXP-289 authorization requires authoritative geometry-bound prep")
    if (
        prep_authority.get("schema") != GEOMETRY_SCHEMA
        or prep_authority.get("authority_scope") != GEOMETRY_SCOPE
        or prep_authority.get("manifest_digest") != expected_geometry_digest
        or prep_authority.get("scientific_execution_digest") != scientific_digest
        or prep_authority.get("authoritative_execution_digest") != outer_digest
        or prep_authority.get("geometry_configuration_match") is not True
        or prep_authority.get("confirmatory_authority") is not False
    ):
        raise ValueError("EXP-289 authorization prep/geometry lineage mismatch")

    confirmatory_n = (prep_artifact.get("sample_size_freeze") or {}).get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not MIN_N <= confirmatory_n <= MAX_N:
        raise ValueError("EXP-289 authorization requires confirmatory_n within frozen bounds")

    checkpoint_errors = _checkpoint_errors(
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        scientific_execution=scientific,
        geometry_authority=authority,
        outer_execution_digest=outer_digest,
        expected_geometry_digest=expected_geometry_digest,
        expected_geometry_configuration=expected_geometry_configuration,
        execution_code_digest=execution_code_digest,
    )
    if checkpoint_errors:
        raise ValueError("; ".join(checkpoint_errors))

    lineage = {
        "protocol_digest": protocol_digest,
        "development_geometry_digest": expected_geometry_digest,
        "development_geometry_configuration_digest": canonical_sha256(expected_geometry_configuration),
        "development_execution_digest": outer_digest,
        "scientific_execution_digest": scientific_digest,
        "prep_digest": prep_artifact.get("prep_digest"),
        "execution_code_digest": execution_code_digest,
        "challenge_contract_digest": challenge_contract_digest(),
        "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
        "checkpoint_execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
        "checkpoint_no_nogood_final_digest": checkpoint_receipt.get("no_nogood_final_digest"),
        "checkpoint_local_nogood_final_digest": checkpoint_receipt.get("local_nogood_final_digest"),
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_n": confirmatory_n,
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "lineage": lineage,
        "preflight": {
            "development_artifact_valid": True,
            "prep_artifact_valid": True,
            "prep_ready": True,
            "authoritative_geometry_bound": True,
            "geometry_configuration_bound": True,
            "development_prep_lineage_bound": True,
            "protocol_lineage_bound": True,
            "code_lineage_bound": True,
            "challenge_contract_bound": True,
            "checkpoint_receipt_valid": True,
            "checkpoint_file_hash_bound": True,
            "checkpoint_execution_contract_bound": True,
            "checkpoint_functional_state_verified": True,
            "pre_beacon_boundary_preserved": True,
            "all_checks_passed": True,
        },
        "authorization_digest": "",
    }
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp289_confirmatory_execution_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-289 confirmatory execution authorization: " + "; ".join(errors))
    return payload
