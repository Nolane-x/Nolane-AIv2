from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp286_confirmatory_prep import validate_exp286_confirmatory_prep
from .exp286_paired_runner import validate_exp286_paired_development


SCHEMA = "NLM-EXP-286-CONFIRMATORY-EXECUTION-AUTH-V1"
EXPERIMENT_ID = "EXP-286"
GEOMETRY_SCHEMA = "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1"
GEOMETRY_SCOPE = "DEVELOPMENT_PILOT_ONLY"


def _authorization_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def _require_authoritative_geometry(
    execution_artifact: dict[str, Any],
    *,
    expected_geometry_digest: str,
) -> None:
    authority = execution_artifact.get("development_geometry_authority") or {}
    if (
        authority.get("schema") != GEOMETRY_SCHEMA
        or authority.get("authority_scope") != GEOMETRY_SCOPE
        or authority.get("manifest_digest") != expected_geometry_digest
        or authority.get("confirmatory_authority") is not False
    ):
        raise ValueError("EXP-286 authorization requires authoritative DEVELOPMENT geometry")


def validate_exp286_confirmatory_execution_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-286 confirmatory execution authorization schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-286 confirmatory execution authorization experiment id drift")
    if payload.get("status") != "AUTHORIZED_NOT_EXECUTED":
        errors.append("EXP-286 authorization must remain pre-execution")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-286 authorization cannot promote scientific evidence")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("EXP-286 authorization cannot consume confirmatory data")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-286 authorization cannot materialize confirmatory seed entropy")
    if payload.get("challenge_materialized") is not False:
        errors.append("EXP-286 authorization cannot materialize challenge data")
    if "seeds" in payload or "challenge_seed" in payload or "beacon" in payload:
        errors.append("EXP-286 authorization payload cannot contain confirmatory entropy")

    confirmatory_n = payload.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not 32 <= confirmatory_n <= 128:
        errors.append("EXP-286 authorization confirmatory_n is outside frozen bounds")

    lineage = payload.get("lineage") or {}
    for key in (
        "development_geometry_digest",
        "development_execution_digest",
        "prep_digest",
        "protocol_digest",
        "execution_code_digest",
    ):
        value = lineage.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"EXP-286 authorization lineage is missing {key}")

    preflight = payload.get("preflight") or {}
    expected_checks = {
        "development_artifact_valid": True,
        "prep_artifact_valid": True,
        "prep_ready": True,
        "authoritative_geometry_bound": True,
        "development_prep_lineage_bound": True,
        "protocol_lineage_bound": True,
        "pre_beacon_boundary_preserved": True,
    }
    for key, expected in expected_checks.items():
        if preflight.get(key) is not expected:
            errors.append(f"EXP-286 authorization preflight failed: {key}")
    if preflight.get("all_checks_passed") is not True:
        errors.append("EXP-286 authorization preflight is not fully closed")

    digest = payload.get("authorization_digest")
    if digest not in (None, "") and digest != _authorization_digest(payload):
        errors.append("EXP-286 authorization digest mismatch")
    return errors


def authorize_exp286_confirmatory_execution(
    *,
    execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    expected_geometry_digest: str,
    execution_code_digest: str,
) -> dict[str, Any]:
    development_errors = validate_exp286_paired_development(execution_artifact)
    if development_errors:
        raise ValueError("invalid EXP-286 development artifact: " + "; ".join(development_errors))
    prep_errors = validate_exp286_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-286 confirmatory prep: " + "; ".join(prep_errors))
    if not expected_geometry_digest:
        raise ValueError("expected_geometry_digest is required")
    if not execution_code_digest:
        raise ValueError("execution_code_digest is required")

    _require_authoritative_geometry(
        execution_artifact,
        expected_geometry_digest=expected_geometry_digest,
    )

    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED" or prep_artifact.get("confirmatory_ready") is not True:
        raise ValueError("EXP-286 authorization requires a ready Gate A prep artifact")
    if prep_artifact.get("evidence_level") != "EV-E2" or prep_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-286 authorization requires EV-E2 / UNVERIFIED prep evidence")
    if execution_artifact.get("evidence_level") != "EV-E2" or execution_artifact.get("decision") != "UNVERIFIED":
        raise ValueError("EXP-286 authorization requires EV-E2 / UNVERIFIED development evidence")

    execution_digest = execution_artifact.get("artifact_digest")
    prep_execution_digest = prep_artifact.get("development_execution_digest")
    if not execution_digest or prep_execution_digest != execution_digest:
        raise ValueError("EXP-286 authorization development/prep lineage mismatch")
    if prep_artifact.get("protocol_digest") != execution_artifact.get("protocol_digest"):
        raise ValueError("EXP-286 authorization protocol lineage mismatch")

    freeze = prep_artifact.get("sample_size_freeze") or {}
    confirmatory_n = freeze.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not 32 <= confirmatory_n <= 128:
        raise ValueError("EXP-286 authorization requires confirmatory_n within frozen bounds")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUTHORIZED_NOT_EXECUTED",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "confirmatory_n": confirmatory_n,
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "lineage": {
            "development_geometry_digest": expected_geometry_digest,
            "development_execution_digest": execution_digest,
            "prep_digest": prep_artifact.get("prep_digest"),
            "protocol_digest": execution_artifact.get("protocol_digest"),
            "execution_code_digest": execution_code_digest,
        },
        "preflight": {
            "development_artifact_valid": True,
            "prep_artifact_valid": True,
            "prep_ready": True,
            "authoritative_geometry_bound": True,
            "development_prep_lineage_bound": True,
            "protocol_lineage_bound": True,
            "pre_beacon_boundary_preserved": True,
            "all_checks_passed": True,
        },
        "authorization_digest": "",
    }
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp286_confirmatory_execution_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-286 confirmatory execution authorization: " + "; ".join(errors))
    return payload
