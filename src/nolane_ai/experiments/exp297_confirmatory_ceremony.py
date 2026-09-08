from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from nolane_ai.experiments.exp297_challenge_worlds import challenge_contract_digest
from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    validate_exp297_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    validate_exp297_confirmatory_prep,
)
from nolane_ai.experiments.exp297_paired_runner import validate_exp297_execution
from nolane_ai.experiments.exp297_reconstruction_court import (
    validate_exp297_confirmatory_reconstruction,
)
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import require_frozen_stage_a_v1_sha256

SCHEMA = "NLM-EXP-297-CONFIRMATORY-GATE-A-SEAL-V1"
EXPERIMENT_ID = "EXP-297"
STATUS = "CONFIRMATORY_GATE_A_SEALED"


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
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "lineage": payload.get("lineage"),
            "execution_authorization_digest": (
                payload.get("execution_authorization") or {}
            ).get("authorization_digest"),
            "reconstruction_digest": (
                payload.get("reconstruction_authorization") or {}
            ).get("reconstruction_digest"),
        }
    )


def _code_tree_errors(
    *,
    code_tree_digest: str,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not _valid_hex_digest(code_tree_digest, {64}):
        return ["EXP-297 code-tree digest must be a 64-hex SHA-256 digest"]
    prep_lineage = prep_artifact.get("lineage") or {}
    auth_lineage = execution_authorization.get("lineage") or {}
    auth_codes = execution_authorization.get("code_digests") or {}
    observed = {
        "development_execution.code_digest": development_execution_artifact.get(
            "code_digest"
        ),
        "prep.analysis_code_digest": prep_lineage.get("analysis_code_digest"),
        "authorization.development_code_digest": auth_lineage.get(
            "development_code_digest"
        ),
        "authorization.evaluator_code_digest": auth_codes.get("evaluator"),
        "authorization.execution_code_digest": auth_codes.get("execution"),
        "reconstruction.reconstruction_code_digest": reconstruction_authorization.get(
            "reconstruction_code_digest"
        ),
    }
    for label, value in observed.items():
        if value != code_tree_digest:
            errors.append(f"EXP-297 code-tree closure mismatch: {label}")
    return errors


def _lineage_from_inputs(
    *,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
) -> dict[str, Any]:
    auth_lineage = execution_authorization.get("lineage") or {}
    return {
        "protocol_digest": development_execution_artifact.get("protocol_digest"),
        "development_execution_digest": development_execution_artifact.get(
            "artifact_digest"
        ),
        "development_code_digest": development_execution_artifact.get("code_digest"),
        "arm_registry_digest": (prep_artifact.get("lineage") or {}).get(
            "arm_registry_digest"
        ),
        "prep_digest": prep_artifact.get("prep_digest"),
        "execution_authorization_digest": execution_authorization.get(
            "authorization_digest"
        ),
        "reconstruction_digest": reconstruction_authorization.get(
            "reconstruction_digest"
        ),
        "pair_audit_digest": auth_lineage.get("pair_audit_digest"),
        "model_init_seed": auth_lineage.get("model_init_seed"),
        "frozen_analysis_digest": execution_authorization.get(
            "frozen_analysis_digest"
        ),
        "sample_size_freeze_digest": execution_authorization.get(
            "sample_size_freeze_digest"
        ),
    }


def seal_exp297_confirmatory_gate_a(
    *,
    protocol_digest: str,
    development_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    code_tree_digest: str,
    freeze_commit_sha: str,
    freeze_commit_timestamp_utc: str,
) -> dict[str, Any]:
    require_frozen_stage_a_v1_sha256(protocol_digest)
    if not _valid_hex_digest(freeze_commit_sha, {40, 64}):
        raise ValueError("EXP-297 freeze commit SHA must be a 40- or 64-hex digest")
    try:
        _parse_utc(freeze_commit_timestamp_utc)
    except ValueError as exc:
        raise ValueError(f"EXP-297 freeze commit timestamp invalid: {exc}") from exc

    execution_errors = validate_exp297_execution(development_execution_artifact)
    if execution_errors:
        raise ValueError(
            "invalid EXP-297 DEVELOPMENT execution: " + "; ".join(execution_errors)
        )
    prep_errors = validate_exp297_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid EXP-297 confirmatory prep: " + "; ".join(prep_errors))
    auth_errors = validate_exp297_confirmatory_execution_authorization(
        execution_authorization
    )
    if auth_errors:
        raise ValueError(
            "invalid EXP-297 execution authorization: " + "; ".join(auth_errors)
        )
    reconstruction_errors = validate_exp297_confirmatory_reconstruction(
        reconstruction_authorization
    )
    if reconstruction_errors:
        raise ValueError(
            "invalid EXP-297 reconstruction authorization: "
            + "; ".join(reconstruction_errors)
        )

    if development_execution_artifact.get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-297 ceremony protocol/development lineage mismatch")
    if (prep_artifact.get("lineage") or {}).get("protocol_digest") != protocol_digest:
        raise ValueError("EXP-297 ceremony protocol/prep lineage mismatch")
    if (execution_authorization.get("lineage") or {}).get(
        "protocol_digest"
    ) != protocol_digest:
        raise ValueError("EXP-297 ceremony protocol/authorization lineage mismatch")
    if (reconstruction_authorization.get("lineage") or {}).get(
        "protocol_digest"
    ) != protocol_digest:
        raise ValueError("EXP-297 ceremony protocol/reconstruction lineage mismatch")

    code_errors = _code_tree_errors(
        code_tree_digest=code_tree_digest,
        development_execution_artifact=development_execution_artifact,
        prep_artifact=prep_artifact,
        execution_authorization=execution_authorization,
        reconstruction_authorization=reconstruction_authorization,
    )
    if code_errors:
        raise ValueError("EXP-297 code-tree closure failed: " + "; ".join(code_errors))

    if execution_authorization.get("challenge_contract_digest") != challenge_contract_digest():
        raise ValueError("EXP-297 ceremony challenge contract drift")
    if reconstruction_authorization.get(
        "challenge_contract_digest"
    ) != challenge_contract_digest():
        raise ValueError("EXP-297 reconstruction challenge contract drift")
    if reconstruction_authorization.get("execution_authorization") != execution_authorization:
        raise ValueError("EXP-297 ceremony reconstruction/authorization binding mismatch")
    if execution_authorization.get("lineage", {}).get("prep_digest") != prep_artifact.get(
        "prep_digest"
    ):
        raise ValueError("EXP-297 ceremony prep/authorization digest mismatch")
    if reconstruction_authorization.get("lineage", {}).get(
        "prep_digest"
    ) != prep_artifact.get("prep_digest"):
        raise ValueError("EXP-297 ceremony prep/reconstruction digest mismatch")

    confirmatory_n = execution_authorization.get("confirmatory_n")
    reserved = deepcopy(execution_authorization.get("reserved_replicate_ids") or [])
    if reconstruction_authorization.get("confirmatory_n") != confirmatory_n:
        raise ValueError("EXP-297 ceremony confirmatory n mismatch")
    if reconstruction_authorization.get("reserved_replicate_ids") != reserved:
        raise ValueError("EXP-297 ceremony reserved replicate lineage mismatch")

    lineage = _lineage_from_inputs(
        development_execution_artifact=development_execution_artifact,
        prep_artifact=prep_artifact,
        execution_authorization=execution_authorization,
        reconstruction_authorization=reconstruction_authorization,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": STATUS,
        "confirmatory_ready": True,
        "confirmatory_ready_scope": "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY",
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "semantic_authority_promoted": False,
        "freeze_commit_sha": freeze_commit_sha,
        "freeze_commit_timestamp_utc": freeze_commit_timestamp_utc,
        "code_tree_digest": code_tree_digest,
        "challenge_contract_digest": challenge_contract_digest(),
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "model_geometry": deepcopy(execution_authorization.get("model_geometry") or {}),
        "court_ceiling": execution_authorization.get("court_ceiling"),
        "lineage": lineage,
        "execution_authorization": deepcopy(execution_authorization),
        "reconstruction_authorization": deepcopy(reconstruction_authorization),
        "development_baseline": {
            "artifact_digest": development_execution_artifact.get("artifact_digest"),
            "code_digest": development_execution_artifact.get("code_digest"),
            "evidence_level": development_execution_artifact.get("evidence_level"),
            "decision": development_execution_artifact.get("decision"),
        },
        "pre_beacon_binding_digest": "",
        "seal_digest": "",
    }
    payload["pre_beacon_binding_digest"] = _pre_beacon_binding(payload)
    payload["seal_digest"] = _seal_digest(payload)
    errors = validate_exp297_confirmatory_gate_a_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 Gate A seal: " + "; ".join(errors))
    return payload


def validate_exp297_confirmatory_gate_a_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 Gate A seal identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 Gate A seal cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-297 Gate A seal status drift")
    if payload.get("confirmatory_ready") is not True:
        errors.append("EXP-297 Gate A seal must record narrow machinery readiness")
    if payload.get("confirmatory_ready_scope") != "FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY":
        errors.append("EXP-297 Gate A readiness scope drift")
    for flag in (
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
        "semantic_authority_promoted",
    ):
        if payload.get(flag) is not False:
            errors.append(f"EXP-297 Gate A forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-297 Gate A cannot materialize seeds")

    rendered = repr(payload).lower()
    for forbidden in ("beacon_receipt", "challenge_seed", "challenge_candidates"):
        if forbidden in rendered:
            errors.append(f"EXP-297 Gate A seal contains forbidden pre-beacon material: {forbidden}")

    protocol_digest = (payload.get("lineage") or {}).get("protocol_digest")
    try:
        require_frozen_stage_a_v1_sha256(protocol_digest)
    except ValueError as exc:
        errors.append(str(exc))
    if not _valid_hex_digest(payload.get("freeze_commit_sha"), {40, 64}):
        errors.append("EXP-297 Gate A freeze commit SHA invalid")
    try:
        _parse_utc(payload.get("freeze_commit_timestamp_utc"))
    except ValueError:
        errors.append("EXP-297 Gate A freeze commit timestamp invalid")
    if not _valid_hex_digest(payload.get("code_tree_digest"), {64}):
        errors.append("EXP-297 Gate A code-tree digest invalid")
    if payload.get("challenge_contract_digest") != challenge_contract_digest():
        errors.append("EXP-297 Gate A challenge contract digest mismatch")
    if payload.get("seal_digest") != _seal_digest(payload):
        errors.append("EXP-297 Gate A seal digest mismatch")
    if payload.get("pre_beacon_binding_digest") != _pre_beacon_binding(payload):
        errors.append("EXP-297 Gate A pre-beacon binding digest mismatch")

    auth = payload.get("execution_authorization")
    reconstruction = payload.get("reconstruction_authorization")
    if not isinstance(auth, dict):
        errors.append("EXP-297 Gate A execution authorization missing")
        return errors
    if not isinstance(reconstruction, dict):
        errors.append("EXP-297 Gate A reconstruction authorization missing")
        return errors
    auth_errors = validate_exp297_confirmatory_execution_authorization(auth)
    if auth_errors:
        errors.append("embedded EXP-297 execution authorization invalid: " + "; ".join(auth_errors))
        return errors
    reconstruction_errors = validate_exp297_confirmatory_reconstruction(reconstruction)
    if reconstruction_errors:
        errors.append("embedded EXP-297 reconstruction authorization invalid: " + "; ".join(reconstruction_errors))
        return errors
    if reconstruction.get("execution_authorization") != auth:
        errors.append("EXP-297 Gate A reconstruction/authorization binding mismatch")

    code_tree_digest = payload.get("code_tree_digest")
    auth_lineage = auth.get("lineage") or {}
    auth_codes = auth.get("code_digests") or {}
    for label, value in {
        "development": auth_lineage.get("development_code_digest"),
        "analysis": auth_lineage.get("analysis_code_digest"),
        "evaluator": auth_codes.get("evaluator"),
        "execution": auth_codes.get("execution"),
        "reconstruction": reconstruction.get("reconstruction_code_digest"),
    }.items():
        if value != code_tree_digest:
            errors.append(f"EXP-297 Gate A code-tree closure mismatch: {label}")

    if payload.get("confirmatory_n") != auth.get("confirmatory_n"):
        errors.append("EXP-297 Gate A confirmatory n mismatch")
    if payload.get("reserved_replicate_ids") != auth.get("reserved_replicate_ids"):
        errors.append("EXP-297 Gate A reserved replicate lineage mismatch")
    if payload.get("model_geometry") != auth.get("model_geometry"):
        errors.append("EXP-297 Gate A model geometry mismatch")
    if payload.get("court_ceiling") != auth.get("court_ceiling"):
        errors.append("EXP-297 Gate A court ceiling mismatch")

    lineage = payload.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": auth_lineage.get("protocol_digest"),
        "development_execution_digest": auth_lineage.get("development_execution_digest"),
        "development_code_digest": auth_lineage.get("development_code_digest"),
        "arm_registry_digest": auth_lineage.get("arm_registry_digest"),
        "prep_digest": auth_lineage.get("prep_digest"),
        "execution_authorization_digest": auth.get("authorization_digest"),
        "reconstruction_digest": reconstruction.get("reconstruction_digest"),
        "pair_audit_digest": auth_lineage.get("pair_audit_digest"),
        "model_init_seed": auth_lineage.get("model_init_seed"),
        "frozen_analysis_digest": auth.get("frozen_analysis_digest"),
        "sample_size_freeze_digest": auth.get("sample_size_freeze_digest"),
    }
    if lineage != expected_lineage:
        errors.append("EXP-297 Gate A lineage mismatch")

    baseline = payload.get("development_baseline") or {}
    if baseline.get("artifact_digest") != auth_lineage.get("development_execution_digest"):
        errors.append("EXP-297 Gate A DEVELOPMENT baseline digest mismatch")
    if baseline.get("code_digest") != code_tree_digest:
        errors.append("EXP-297 Gate A DEVELOPMENT code-tree binding mismatch")
    if baseline.get("evidence_level") != "EV-E2" or baseline.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 Gate A DEVELOPMENT baseline must remain EV-E2 / UNVERIFIED")
    return errors
