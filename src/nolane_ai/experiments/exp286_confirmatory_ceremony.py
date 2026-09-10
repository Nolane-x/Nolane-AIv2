from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from nolane_ai.experiments.exp286_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp286_confirmatory_authorization import (
    validate_exp286_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp286_confirmatory_prep import (
    validate_exp286_confirmatory_prep,
)
from nolane_ai.experiments.exp286_paired_runner import (
    validate_exp286_paired_development,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_frozen_stage_a_v1_sha256


SCHEMA = "NLM-EXP-286-CONFIRMATORY-GATE-A-SEAL-V1"
EXPERIMENT_ID = "EXP-286"
STATUS = "CONFIRMATORY_GATE_A_SEALED"
READY_SCOPE = "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY"


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


def _pre_beacon_binding(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "code_tree_digest": payload.get("code_tree_digest"),
            "challenge_contract_digest": payload.get("challenge_contract_digest"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "lineage": payload.get("lineage"),
            "execution_authorization_digest": (
                payload.get("execution_authorization") or {}
            ).get("authorization_digest"),
        }
    )


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


def _code_tree_errors(
    *,
    code_tree_digest: str,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not _valid_hex_digest(code_tree_digest, {64}):
        return ["EXP-286 code-tree digest must be a 64-hex SHA-256 digest"]
    auth_lineage = execution_authorization.get("lineage") or {}
    observed = {
        "development_execution.code_digest": development_execution_artifact.get("code_digest"),
        "prep.analysis_code_digest": prep_artifact.get("analysis_code_digest"),
        "authorization.execution_code_digest": auth_lineage.get("execution_code_digest"),
    }
    for label, value in observed.items():
        if value != code_tree_digest:
            errors.append(f"EXP-286 code-tree closure mismatch: {label}")
    return errors


def _lineage_from_inputs(
    *,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    expected_geometry_digest: str,
) -> dict[str, Any]:
    auth_lineage = execution_authorization.get("lineage") or {}
    return {
        "protocol_digest": development_execution_artifact.get("protocol_digest"),
        "development_geometry_digest": expected_geometry_digest,
        "development_execution_digest": development_execution_artifact.get("artifact_digest"),
        "development_code_digest": development_execution_artifact.get("code_digest"),
        "prep_digest": prep_artifact.get("prep_digest"),
        "analysis_code_digest": prep_artifact.get("analysis_code_digest"),
        "execution_authorization_digest": execution_authorization.get("authorization_digest"),
        "execution_code_digest": auth_lineage.get("execution_code_digest"),
        "challenge_contract_digest": auth_lineage.get("challenge_contract_digest"),
    }


def seal_exp286_confirmatory_gate_a(
    *,
    protocol_digest: str,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    expected_geometry_digest: str,
    code_tree_digest: str,
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
) -> dict[str, Any]:
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if not _valid_hex_digest(freeze_commit_sha, {40, 64}):
        raise ValueError("EXP-286 freeze commit SHA must be a 40- or 64-hex digest")
    try:
        _parse_utc(freeze_commit_timestamp_utc)
    except ValueError as exc:
        raise ValueError(f"EXP-286 freeze commit timestamp invalid: {exc}") from exc
    if not _valid_hex_digest(expected_geometry_digest, {64}):
        raise ValueError("EXP-286 DEVELOPMENT geometry digest must be a 64-hex SHA-256 digest")

    execution_errors = validate_exp286_paired_development(development_execution_artifact)
    if execution_errors:
        raise ValueError(
            "invalid EXP-286 DEVELOPMENT execution: " + "; ".join(execution_errors)
        )
    prep_errors = validate_exp286_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-286 confirmatory prep: " + "; ".join(prep_errors))
    authorization_errors = validate_exp286_confirmatory_execution_authorization(
        execution_authorization
    )
    if authorization_errors:
        raise ValueError(
            "invalid EXP-286 execution authorization: "
            + "; ".join(authorization_errors)
        )

    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep_artifact.get(
        "confirmatory_ready"
    ) is not True:
        raise ValueError("EXP-286 Gate-A seal requires a ready confirmatory prep")
    if development_execution_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 ceremony protocol/development lineage mismatch")
    if prep_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 ceremony protocol/prep lineage mismatch")

    auth_lineage = execution_authorization.get("lineage") or {}
    if auth_lineage.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-286 ceremony protocol/authorization lineage mismatch")

    authority = development_execution_artifact.get("development_geometry_authority") or {}
    if (
        authority.get("schema") != "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1"
        or authority.get("authority_scope") != "DEVELOPMENT_PILOT_ONLY"
        or authority.get("manifest_digest") != expected_geometry_digest
        or authority.get("confirmatory_authority") is not False
        or auth_lineage.get("development_geometry_digest") != expected_geometry_digest
    ):
        raise ValueError("EXP-286 ceremony DEVELOPMENT geometry authority mismatch")

    execution_digest = development_execution_artifact.get("artifact_digest")
    if not execution_digest or prep_artifact.get("development_execution_digest") != execution_digest:
        raise ValueError("EXP-286 ceremony development/prep lineage mismatch")
    if auth_lineage.get("development_execution_digest") != execution_digest:
        raise ValueError("EXP-286 ceremony development/authorization lineage mismatch")
    if auth_lineage.get("prep_digest") != prep_artifact.get("prep_digest"):
        raise ValueError("EXP-286 ceremony prep/authorization lineage mismatch")

    current_challenge_digest = challenge_contract_digest()
    if auth_lineage.get("challenge_contract_digest") != current_challenge_digest:
        raise ValueError("EXP-286 ceremony challenge contract drift")

    code_errors = _code_tree_errors(
        code_tree_digest=code_tree_digest,
        development_execution_artifact=development_execution_artifact,
        prep_artifact=prep_artifact,
        execution_authorization=execution_authorization,
    )
    if code_errors:
        raise ValueError("EXP-286 code-tree closure failed: " + "; ".join(code_errors))

    confirmatory_n = execution_authorization.get("confirmatory_n")
    prep_n = (prep_artifact.get("sample_size_freeze") or {}).get("confirmatory_n")
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
        or prep_n != confirmatory_n
    ):
        raise ValueError("EXP-286 ceremony confirmatory sample-size lineage mismatch")

    lineage = _lineage_from_inputs(
        development_execution_artifact=development_execution_artifact,
        prep_artifact=prep_artifact,
        execution_authorization=execution_authorization,
        expected_geometry_digest=expected_geometry_digest,
    )
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
        "challenge_contract_digest": current_challenge_digest,
        "confirmatory_n": confirmatory_n,
        "lineage": lineage,
        "execution_authorization": deepcopy(execution_authorization),
        "pre_beacon_binding_digest": "",
        "seal_digest": "",
    }
    payload["pre_beacon_binding_digest"] = _pre_beacon_binding(payload)
    payload["seal_digest"] = _seal_digest(payload)
    errors = validate_exp286_confirmatory_gate_a_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-286 Gate-A seal: " + "; ".join(errors))
    return payload


def validate_exp286_confirmatory_gate_a_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-286 Gate-A seal must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-286 Gate-A seal identity")
    if payload.get("status") != STATUS:
        errors.append("EXP-286 Gate-A seal status drift")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-286 Gate-A seal cannot promote scientific evidence")
    if payload.get("confirmatory_ready") is not True:
        errors.append("EXP-286 Gate-A seal must record narrow machinery readiness")
    if payload.get("confirmatory_ready_scope") != READY_SCOPE:
        errors.append("EXP-286 Gate-A readiness scope drift")
    for field in (
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(field) is not False:
            errors.append(f"EXP-286 Gate-A seal requires {field}=false")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-286 Gate-A seal cannot materialize challenge seeds")

    forbidden = _forbidden_material_present(payload)
    for item in forbidden:
        errors.append(f"EXP-286 Gate-A seal contains forbidden pre-beacon material: {item}")

    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        errors.append("EXP-286 Gate-A seal lineage missing")
        return errors

    protocol_digest = lineage.get("protocol_digest")
    try:
        require_frozen_stage_a_v1_sha256(protocol_digest)
    except ValueError as exc:
        errors.append(str(exc))

    if not _valid_hex_digest(payload.get("freeze_commit_sha"), {40, 64}):
        errors.append("EXP-286 Gate-A freeze commit SHA invalid")
    try:
        _parse_utc(payload.get("freeze_commit_timestamp_utc"))
    except ValueError:
        errors.append("EXP-286 Gate-A freeze commit timestamp invalid")
    if not _valid_hex_digest(payload.get("code_tree_digest"), {64}):
        errors.append("EXP-286 Gate-A code-tree digest invalid")
    if not _valid_hex_digest(lineage.get("development_geometry_digest"), {64}):
        errors.append("EXP-286 Gate-A DEVELOPMENT geometry digest invalid")

    current_challenge_digest = challenge_contract_digest()
    if payload.get("challenge_contract_digest") != current_challenge_digest:
        errors.append("EXP-286 Gate-A challenge contract digest mismatch")
    if lineage.get("challenge_contract_digest") != current_challenge_digest:
        errors.append("EXP-286 Gate-A lineage challenge contract digest mismatch")

    code_tree_digest = payload.get("code_tree_digest")
    for label, value in {
        "development": lineage.get("development_code_digest"),
        "analysis": lineage.get("analysis_code_digest"),
        "execution": lineage.get("execution_code_digest"),
    }.items():
        if value != code_tree_digest:
            errors.append(f"EXP-286 Gate-A code-tree closure mismatch: {label}")

    auth = payload.get("execution_authorization")
    if not isinstance(auth, dict):
        errors.append("EXP-286 Gate-A execution authorization missing")
        return errors
    auth_errors = validate_exp286_confirmatory_execution_authorization(auth)
    if auth_errors:
        errors.append(
            "embedded EXP-286 execution authorization invalid: " + "; ".join(auth_errors)
        )
        return errors

    auth_lineage = auth.get("lineage") or {}
    expected_pairs = {
        "protocol_digest": auth_lineage.get("protocol_digest"),
        "development_geometry_digest": auth_lineage.get("development_geometry_digest"),
        "development_execution_digest": auth_lineage.get("development_execution_digest"),
        "prep_digest": auth_lineage.get("prep_digest"),
        "execution_code_digest": auth_lineage.get("execution_code_digest"),
        "challenge_contract_digest": auth_lineage.get("challenge_contract_digest"),
    }
    for key, expected in expected_pairs.items():
        if lineage.get(key) != expected:
            errors.append(f"EXP-286 Gate-A lineage digest mismatch: {key}")
    if lineage.get("execution_authorization_digest") != auth.get("authorization_digest"):
        errors.append("EXP-286 Gate-A execution authorization digest mismatch")

    confirmatory_n = payload.get("confirmatory_n")
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
        or confirmatory_n != auth.get("confirmatory_n")
    ):
        errors.append("EXP-286 Gate-A confirmatory sample-size mismatch")

    if payload.get("pre_beacon_binding_digest") != _pre_beacon_binding(payload):
        errors.append("EXP-286 Gate-A pre-beacon binding digest mismatch")
    if payload.get("seal_digest") != _seal_digest(payload):
        errors.append("EXP-286 Gate-A seal digest mismatch")
    return errors
