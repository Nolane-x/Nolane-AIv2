from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_frozen_stage_a_v1_sha256
from .exp289_challenge_worlds import challenge_contract_digest
from .exp289_checkpoint import validate_exp289_checkpoint_receipt
from .exp289_confirmatory_authorization import validate_exp289_confirmatory_execution_authorization
from .exp289_confirmatory_prep import validate_exp289_confirmatory_prep
from .exp289_development_geometry import AUTHORITY_SCOPE as GEOMETRY_SCOPE, SCHEMA as GEOMETRY_SCHEMA
from .exp289_paired_runner import _artifact_digest, validate_exp289_paired_development


SCHEMA = "NLM-EXP-289-CONFIRMATORY-GATE-A-SEAL-V1"
EXPERIMENT_ID = "EXP-289"
STATUS = "CONFIRMATORY_GATE_A_SEALED"
READY_SCOPE = "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY"
MIN_N = 32
MAX_N = 128


def _seal_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("seal_digest", None)
    return canonical_sha256(clean)


def _valid_hex_digest(value: Any, lengths: set[int]) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("freeze commit timestamp must be a non-empty string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("freeze commit timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("freeze commit timestamp must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _forbidden_material_present(payload: dict[str, Any]) -> list[str]:
    rendered = repr(payload).lower()
    forbidden = (
        "beacon_receipt",
        "entropy_hex",
        "challenge_seed",
        "challenge_candidates",
        "confirmatory_observations",
    )
    return [item for item in forbidden if item in rendered]


def _pre_beacon_binding(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "code_tree_digest": payload.get("code_tree_digest"),
            "challenge_contract_digest": payload.get("challenge_contract_digest"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "lineage": payload.get("lineage"),
            "execution_authorization_digest": (payload.get("execution_authorization") or {}).get("authorization_digest"),
        }
    )


def _scientific_execution(
    development_execution_artifact: dict[str, Any],
    *,
    expected_geometry_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    authority = development_execution_artifact.get("development_geometry_authority")
    if not isinstance(authority, dict):
        raise ValueError("EXP-289 Gate-A seal requires authoritative DEVELOPMENT geometry")
    if (
        authority.get("schema") != GEOMETRY_SCHEMA
        or authority.get("authority_scope") != GEOMETRY_SCOPE
        or authority.get("manifest_digest") != expected_geometry_digest
        or authority.get("confirmatory_authority") is not False
    ):
        raise ValueError("EXP-289 Gate-A seal DEVELOPMENT geometry authority mismatch")

    outer_digest = development_execution_artifact.get("artifact_digest")
    if not _valid_hex_digest(outer_digest, {64}) or outer_digest != _artifact_digest(development_execution_artifact):
        raise ValueError("EXP-289 Gate-A seal authoritative DEVELOPMENT self-hash mismatch")

    scientific = deepcopy(development_execution_artifact)
    scientific.pop("development_geometry_authority", None)
    scientific["artifact_digest"] = _artifact_digest(scientific)
    scientific_digest = scientific["artifact_digest"]
    if authority.get("scientific_execution_digest") != scientific_digest:
        raise ValueError("EXP-289 Gate-A seal scientific execution digest mismatch")

    errors = validate_exp289_paired_development(scientific)
    if errors:
        raise ValueError("invalid EXP-289 DEVELOPMENT execution for Gate-A seal: " + "; ".join(errors))
    return scientific, deepcopy(authority), str(outer_digest), str(scientific_digest)


def _checkpoint_lineage_errors(
    *,
    checkpoint_receipt: dict[str, Any],
    authorization_lineage: dict[str, Any],
    scientific_execution: dict[str, Any],
    geometry_authority: dict[str, Any],
    outer_execution_digest: str,
    expected_geometry_digest: str,
    code_tree_digest: str,
) -> list[str]:
    errors = validate_exp289_checkpoint_receipt(checkpoint_receipt)
    if errors:
        return ["EXP-289 checkpoint receipt invalid: " + "; ".join(errors)]

    contract = checkpoint_receipt.get("execution_contract") or {}
    contract_authority = contract.get("development_geometry_authority") or {}
    expected_contract = {
        "protocol_digest": scientific_execution.get("protocol_digest"),
        "development_code_digest": code_tree_digest,
        "scientific_execution_digest": scientific_execution.get("artifact_digest"),
        "authoritative_execution_digest": outer_execution_digest,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            errors.append(f"EXP-289 checkpoint contract lineage mismatch: {key}")
    if (
        contract_authority.get("schema") != GEOMETRY_SCHEMA
        or contract_authority.get("authority_scope") != GEOMETRY_SCOPE
        or contract_authority.get("manifest_digest") != expected_geometry_digest
        or contract_authority.get("scientific_execution_digest") != geometry_authority.get("scientific_execution_digest")
        or contract_authority.get("confirmatory_authority") is not False
    ):
        errors.append("EXP-289 checkpoint authoritative geometry lineage mismatch")

    receipt_pairs = {
        "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "checkpoint_file_sha256": checkpoint_receipt.get("checkpoint_file_sha256"),
        "checkpoint_execution_contract_digest": checkpoint_receipt.get("execution_contract_digest"),
        "checkpoint_no_nogood_final_digest": checkpoint_receipt.get("no_nogood_final_digest"),
        "checkpoint_local_nogood_final_digest": checkpoint_receipt.get("local_nogood_final_digest"),
    }
    for key, expected in receipt_pairs.items():
        if authorization_lineage.get(key) != expected:
            errors.append(f"EXP-289 checkpoint/authorization lineage mismatch: {key}")
    return errors


def seal_exp289_confirmatory_gate_a(
    *,
    protocol_digest: str,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    checkpoint_receipt: dict[str, Any],
    execution_authorization: dict[str, Any],
    expected_geometry_digest: str,
    code_tree_digest: str,
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
) -> dict[str, Any]:
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if not _valid_hex_digest(expected_geometry_digest, {64}):
        raise ValueError("EXP-289 Gate-A seal geometry digest must be a 64-hex SHA-256 digest")
    if not _valid_hex_digest(code_tree_digest, {64}):
        raise ValueError("EXP-289 Gate-A seal code-tree digest must be a 64-hex SHA-256 digest")
    if not _valid_hex_digest(freeze_commit_sha, {40, 64}):
        raise ValueError("EXP-289 freeze commit SHA must be a 40- or 64-hex digest")
    try:
        _parse_utc(freeze_commit_timestamp_utc)
    except ValueError as exc:
        raise ValueError(f"EXP-289 freeze commit timestamp invalid: {exc}") from exc

    scientific, authority, outer_digest, scientific_digest = _scientific_execution(
        development_execution_artifact,
        expected_geometry_digest=expected_geometry_digest,
    )
    if scientific.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 Gate-A seal protocol/development lineage mismatch")
    if scientific.get("code_digest") != code_tree_digest:
        raise ValueError("EXP-289 Gate-A seal code-tree/development lineage mismatch")

    prep_errors = validate_exp289_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-289 confirmatory prep: " + "; ".join(prep_errors))
    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep_artifact.get("confirmatory_ready") is not True:
        raise ValueError("EXP-289 Gate-A seal requires ready confirmatory prep")
    if prep_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 Gate-A seal protocol/prep lineage mismatch")
    if prep_artifact.get("analysis_code_digest") != code_tree_digest:
        raise ValueError("EXP-289 Gate-A seal code-tree/prep lineage mismatch")
    if prep_artifact.get("development_execution_digest") != scientific_digest:
        raise ValueError("EXP-289 Gate-A seal scientific execution/prep lineage mismatch")

    prep_authority = prep_artifact.get("development_geometry_authority") or {}
    if (
        prep_authority.get("schema") != GEOMETRY_SCHEMA
        or prep_authority.get("authority_scope") != GEOMETRY_SCOPE
        or prep_authority.get("manifest_digest") != expected_geometry_digest
        or prep_authority.get("scientific_execution_digest") != scientific_digest
        or prep_authority.get("authoritative_execution_digest") != outer_digest
        or prep_authority.get("geometry_configuration_match") is not True
        or prep_authority.get("confirmatory_authority") is not False
    ):
        raise ValueError("EXP-289 Gate-A seal prep/geometry lineage mismatch")

    auth_errors = validate_exp289_confirmatory_execution_authorization(execution_authorization)
    if auth_errors:
        raise ValueError("invalid EXP-289 execution authorization: " + "; ".join(auth_errors))
    auth_lineage = execution_authorization.get("lineage") or {}
    if auth_lineage.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-289 Gate-A seal protocol/authorization lineage mismatch")
    if auth_lineage.get("development_geometry_digest") != expected_geometry_digest:
        raise ValueError("EXP-289 Gate-A seal geometry/authorization lineage mismatch")
    if auth_lineage.get("development_execution_digest") != outer_digest:
        raise ValueError("EXP-289 Gate-A seal outer execution/authorization lineage mismatch")
    if auth_lineage.get("scientific_execution_digest") != scientific_digest:
        raise ValueError("EXP-289 Gate-A seal scientific execution/authorization lineage mismatch")
    if auth_lineage.get("prep_digest") != prep_artifact.get("prep_digest"):
        raise ValueError("EXP-289 Gate-A seal prep/authorization lineage mismatch")
    if auth_lineage.get("execution_code_digest") != code_tree_digest:
        raise ValueError("EXP-289 Gate-A seal code-tree/authorization lineage mismatch")
    if auth_lineage.get("challenge_contract_digest") != challenge_contract_digest():
        raise ValueError("EXP-289 Gate-A seal challenge-contract drift")

    checkpoint_errors = _checkpoint_lineage_errors(
        checkpoint_receipt=checkpoint_receipt,
        authorization_lineage=auth_lineage,
        scientific_execution=scientific,
        geometry_authority=authority,
        outer_execution_digest=outer_digest,
        expected_geometry_digest=expected_geometry_digest,
        code_tree_digest=code_tree_digest,
    )
    if checkpoint_errors:
        raise ValueError("; ".join(checkpoint_errors))

    confirmatory_n = execution_authorization.get("confirmatory_n")
    prep_n = (prep_artifact.get("sample_size_freeze") or {}).get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not MIN_N <= confirmatory_n <= MAX_N or prep_n != confirmatory_n:
        raise ValueError("EXP-289 Gate-A seal confirmatory sample-size lineage mismatch")

    lineage = {
        "protocol_digest": protocol_digest,
        "development_geometry_digest": expected_geometry_digest,
        "development_geometry_configuration_digest": auth_lineage.get("development_geometry_configuration_digest"),
        "development_execution_digest": outer_digest,
        "scientific_execution_digest": scientific_digest,
        "development_code_digest": scientific.get("code_digest"),
        "prep_digest": prep_artifact.get("prep_digest"),
        "analysis_code_digest": prep_artifact.get("analysis_code_digest"),
        "execution_authorization_digest": execution_authorization.get("authorization_digest"),
        "execution_code_digest": auth_lineage.get("execution_code_digest"),
        "challenge_contract_digest": auth_lineage.get("challenge_contract_digest"),
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
        "confirmatory_ready": True,
        "confirmatory_ready_scope": READY_SCOPE,
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp_utc": freeze_commit_timestamp_utc,
        "code_tree_digest": code_tree_digest,
        "challenge_contract_digest": challenge_contract_digest(),
        "confirmatory_n": confirmatory_n,
        "lineage": lineage,
        "execution_authorization": deepcopy(execution_authorization),
        "pre_beacon_binding_digest": "",
        "seal_digest": "",
    }
    payload["pre_beacon_binding_digest"] = _pre_beacon_binding(payload)
    payload["seal_digest"] = _seal_digest(payload)
    errors = validate_exp289_confirmatory_gate_a_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-289 Gate-A seal: " + "; ".join(errors))
    return payload


def validate_exp289_confirmatory_gate_a_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-289 Gate-A seal must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-289 Gate-A seal identity")
    if payload.get("status") != STATUS:
        errors.append("EXP-289 Gate-A seal status drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-289 Gate-A seal cannot promote scientific evidence")
    if payload.get("confirmatory_ready") is not True:
        errors.append("EXP-289 Gate-A seal must record narrow machinery readiness")
    if payload.get("confirmatory_ready_scope") != READY_SCOPE:
        errors.append("EXP-289 Gate-A readiness scope drift")
    for field in ("confirmatory_data_consumed", "challenge_materialized", "decision_rule_executed"):
        if payload.get(field) is not False:
            errors.append(f"EXP-289 Gate-A seal requires {field}=false")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-289 Gate-A seal cannot materialize future challenge seed entropy")
    for item in _forbidden_material_present(payload):
        errors.append(f"EXP-289 Gate-A seal contains forbidden pre-beacon material: {item}")

    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        errors.append("EXP-289 Gate-A seal lineage missing")
        return errors
    try:
        require_frozen_stage_a_v1_sha256(lineage.get("protocol_digest"))
    except ValueError as exc:
        errors.append(str(exc))

    if not _valid_hex_digest(payload.get("freeze_commit_sha"), {40, 64}):
        errors.append("EXP-289 Gate-A freeze commit SHA invalid")
    try:
        _parse_utc(payload.get("freeze_commit_timestamp_utc"))
    except ValueError:
        errors.append("EXP-289 Gate-A freeze commit timestamp invalid")
    if not _valid_hex_digest(payload.get("code_tree_digest"), {64}):
        errors.append("EXP-289 Gate-A code-tree digest invalid")

    digest_fields = (
        "development_geometry_digest",
        "development_geometry_configuration_digest",
        "development_execution_digest",
        "scientific_execution_digest",
        "development_code_digest",
        "prep_digest",
        "analysis_code_digest",
        "execution_authorization_digest",
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
        if not _valid_hex_digest(lineage.get(key), {64}):
            errors.append(f"EXP-289 Gate-A lineage invalid/missing digest: {key}")

    current_challenge_digest = challenge_contract_digest()
    if payload.get("challenge_contract_digest") != current_challenge_digest:
        errors.append("EXP-289 Gate-A challenge contract digest mismatch")
    if lineage.get("challenge_contract_digest") != current_challenge_digest:
        errors.append("EXP-289 Gate-A lineage challenge contract digest mismatch")

    code_tree_digest = payload.get("code_tree_digest")
    for label, value in {
        "development": lineage.get("development_code_digest"),
        "analysis": lineage.get("analysis_code_digest"),
        "execution": lineage.get("execution_code_digest"),
    }.items():
        if value != code_tree_digest:
            errors.append(f"EXP-289 Gate-A code-tree closure mismatch: {label}")

    auth = payload.get("execution_authorization")
    if not isinstance(auth, dict):
        errors.append("EXP-289 Gate-A execution authorization missing")
        return errors
    auth_errors = validate_exp289_confirmatory_execution_authorization(auth)
    if auth_errors:
        errors.append("embedded EXP-289 execution authorization invalid: " + "; ".join(auth_errors))
        return errors

    auth_lineage = auth.get("lineage") or {}
    paired_keys = (
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
    for key in paired_keys:
        if lineage.get(key) != auth_lineage.get(key):
            errors.append(f"EXP-289 Gate-A lineage digest mismatch: {key}")
    if lineage.get("execution_authorization_digest") != auth.get("authorization_digest"):
        errors.append("EXP-289 Gate-A execution authorization digest mismatch")

    confirmatory_n = payload.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not MIN_N <= confirmatory_n <= MAX_N or confirmatory_n != auth.get("confirmatory_n"):
        errors.append("EXP-289 Gate-A confirmatory sample-size mismatch")

    if payload.get("pre_beacon_binding_digest") != _pre_beacon_binding(payload):
        errors.append("EXP-289 Gate-A pre-beacon binding digest mismatch")
    if payload.get("seal_digest") != _seal_digest(payload):
        errors.append("EXP-289 Gate-A seal digest mismatch")
    return errors
