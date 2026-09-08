from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.experiments.exp297_confirmatory_execution_court import (
    validate_exp297_confirmatory_execution_authorization,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    validate_exp297_confirmatory_prep,
)
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-297-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
EXPERIMENT_ID = "EXP-297"
STATUS = "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"

_VALIDATED_PREP_DIGESTS: set[str] = set()
_VALIDATED_AUTH_DIGESTS: set[str] = set()


def _reconstruction_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("reconstruction_digest", None)
    return canonical_sha256(clean)


def _prep_content_digest(prep: dict[str, Any]) -> str:
    clean = deepcopy(prep)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def _auth_content_digest(auth: dict[str, Any]) -> str:
    clean = deepcopy(auth)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def _validate_prep_cached(prep: dict[str, Any]) -> list[str]:
    digest = prep.get("prep_digest")
    if not isinstance(digest, str) or not digest or digest != _prep_content_digest(prep):
        return ["EXP-297 reconstruction embedded prep digest mismatch"]
    if digest not in _VALIDATED_PREP_DIGESTS:
        errors = validate_exp297_confirmatory_prep(prep)
        if errors:
            return ["invalid EXP-297 reconstruction prep: " + "; ".join(errors)]
        _VALIDATED_PREP_DIGESTS.add(digest)
    return []


def _validate_auth_cached(auth: dict[str, Any]) -> list[str]:
    digest = auth.get("authorization_digest")
    if not isinstance(digest, str) or not digest or digest != _auth_content_digest(auth):
        return ["EXP-297 reconstruction embedded execution authorization digest mismatch"]
    if digest not in _VALIDATED_AUTH_DIGESTS:
        errors = validate_exp297_confirmatory_execution_authorization(auth)
        if errors:
            return ["invalid EXP-297 execution authorization: " + "; ".join(errors)]
        _VALIDATED_AUTH_DIGESTS.add(digest)
    return []


def _reconstruction_contract(candidate_count: int) -> dict[str, Any]:
    return {
        "candidate_count_per_replicate": candidate_count,
        "reconstruct_beacon_derived_seed": True,
        "reconstruct_challenge_world": True,
        "reconstruct_exact_fidelity_receipt": True,
        "reconstruct_matched_neural_pair": True,
        "reconstruct_neural_and_semantic_costs": True,
        "reconstruct_primary_and_safety_counts": True,
        "candidate_order_must_match": True,
        "truth_and_stratum_evaluator_only": True,
        "court_inconclusive_grants_no_authority": True,
    }


def _binding_digest(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "lineage": payload.get("lineage"),
            "model_geometry": payload.get("model_geometry"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "challenge_contract_digest": payload.get("challenge_contract_digest"),
            "frozen_analysis_digest": payload.get("frozen_analysis_digest"),
            "sample_size_freeze_digest": payload.get("sample_size_freeze_digest"),
            "reconstruction_contract": payload.get("reconstruction_contract"),
            "reconstruction_code_digest": payload.get("reconstruction_code_digest"),
        }
    )


def authorize_exp297_confirmatory_reconstruction(
    *,
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_code_digest: str,
) -> dict[str, Any]:
    prep_errors = _validate_prep_cached(prep_artifact)
    if prep_errors:
        raise ValueError("; ".join(prep_errors))
    auth_errors = _validate_auth_cached(execution_authorization)
    if auth_errors:
        raise ValueError("; ".join(auth_errors))
    if not isinstance(reconstruction_code_digest, str) or not reconstruction_code_digest:
        raise ValueError("reconstruction_code_digest is required")
    if execution_authorization.get("prep_artifact") != prep_artifact:
        raise ValueError("EXP-297 reconstruction prep/authorization binding mismatch")

    candidate_count = execution_authorization.get("candidate_count_per_replicate")
    if candidate_count != 16:
        raise ValueError("EXP-297 reconstruction candidate count drift")
    model_geometry = deepcopy(execution_authorization.get("model_geometry") or {})
    lineage = {
        "prep_digest": prep_artifact.get("prep_digest"),
        "execution_authorization_digest": execution_authorization.get("authorization_digest"),
        "protocol_digest": (execution_authorization.get("lineage") or {}).get("protocol_digest"),
        "development_execution_digest": (execution_authorization.get("lineage") or {}).get("development_execution_digest"),
        "pair_audit_digest": (execution_authorization.get("lineage") or {}).get("pair_audit_digest"),
        "model_init_seed": (execution_authorization.get("lineage") or {}).get("model_init_seed"),
        "challenge_contract_digest": execution_authorization.get("challenge_contract_digest"),
        "reconstruction_code_digest": reconstruction_code_digest,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": STATUS,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "decision_rule_executed": False,
        "semantic_authority_promoted": False,
        "confirmatory_n": execution_authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(execution_authorization.get("reserved_replicate_ids") or []),
        "candidate_count_per_replicate": candidate_count,
        "challenge_contract_digest": execution_authorization.get("challenge_contract_digest"),
        "model_geometry": model_geometry,
        "court_ceiling": model_geometry.get("max_exact_assignments"),
        "frozen_analysis_digest": execution_authorization.get("frozen_analysis_digest"),
        "sample_size_freeze_digest": execution_authorization.get("sample_size_freeze_digest"),
        "reconstruction_contract": _reconstruction_contract(candidate_count),
        "reconstruction_code_digest": reconstruction_code_digest,
        "lineage": lineage,
        "prep_artifact": deepcopy(prep_artifact),
        "execution_authorization": deepcopy(execution_authorization),
        "binding_digest": "",
        "reconstruction_digest": "",
    }
    payload["binding_digest"] = _binding_digest(payload)
    payload["reconstruction_digest"] = _reconstruction_digest(payload)
    errors = validate_exp297_confirmatory_reconstruction(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 reconstruction authorization: " + "; ".join(errors))
    return payload


def build_exp297_confirmatory_reconstruction(
    *,
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_code_digest: str,
) -> dict[str, Any]:
    return authorize_exp297_confirmatory_reconstruction(
        prep_artifact=prep_artifact,
        execution_authorization=execution_authorization,
        reconstruction_code_digest=reconstruction_code_digest,
    )


def validate_exp297_confirmatory_reconstruction(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 reconstruction authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 reconstruction authorization cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-297 reconstruction authorization status drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
        "semantic_authority_promoted",
    ):
        if payload.get(flag) is not False:
            errors.append(f"EXP-297 reconstruction forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-297 reconstruction cannot materialize seeds")
    rendered = repr(payload).lower()
    for forbidden in ("beacon_receipt", "challenge_seed", "challenge_candidates"):
        if forbidden in rendered:
            errors.append(f"EXP-297 reconstruction contains forbidden pre-freeze material: {forbidden}")
    if payload.get("reconstruction_digest") != _reconstruction_digest(payload):
        errors.append("EXP-297 reconstruction authorization digest mismatch")

    prep = payload.get("prep_artifact")
    auth = payload.get("execution_authorization")
    if not isinstance(prep, dict):
        errors.append("EXP-297 reconstruction prep artifact missing")
        return errors
    if not isinstance(auth, dict):
        errors.append("EXP-297 reconstruction execution authorization missing")
        return errors
    prep_errors = _validate_prep_cached(prep)
    if prep_errors:
        errors.extend(prep_errors)
        return errors
    auth_errors = _validate_auth_cached(auth)
    if auth_errors:
        errors.extend(auth_errors)
        return errors
    if auth.get("prep_artifact") != prep:
        errors.append("EXP-297 reconstruction prep/authorization binding mismatch")

    expected_model_geometry = auth.get("model_geometry") or {}
    if payload.get("model_geometry") != expected_model_geometry:
        errors.append("EXP-297 reconstruction model geometry mismatch")
    if payload.get("court_ceiling") != expected_model_geometry.get("max_exact_assignments"):
        errors.append("EXP-297 reconstruction court ceiling mismatch")
    for key in (
        "confirmatory_n",
        "reserved_replicate_ids",
        "candidate_count_per_replicate",
        "challenge_contract_digest",
        "frozen_analysis_digest",
        "sample_size_freeze_digest",
    ):
        if payload.get(key) != auth.get(key):
            errors.append(f"EXP-297 reconstruction {key} mismatch")

    candidate_count = auth.get("candidate_count_per_replicate")
    expected_contract = _reconstruction_contract(candidate_count)
    if payload.get("reconstruction_contract") != expected_contract:
        errors.append("EXP-297 reconstruction contract drift")
    reconstruction_code_digest = payload.get("reconstruction_code_digest")
    if not isinstance(reconstruction_code_digest, str) or not reconstruction_code_digest:
        errors.append("EXP-297 reconstruction code digest missing")
        return errors

    auth_lineage = auth.get("lineage") or {}
    expected_lineage = {
        "prep_digest": prep.get("prep_digest"),
        "execution_authorization_digest": auth.get("authorization_digest"),
        "protocol_digest": auth_lineage.get("protocol_digest"),
        "development_execution_digest": auth_lineage.get("development_execution_digest"),
        "pair_audit_digest": auth_lineage.get("pair_audit_digest"),
        "model_init_seed": auth_lineage.get("model_init_seed"),
        "challenge_contract_digest": auth.get("challenge_contract_digest"),
        "reconstruction_code_digest": reconstruction_code_digest,
    }
    if payload.get("lineage") != expected_lineage:
        errors.append("EXP-297 reconstruction lineage mismatch")
    if payload.get("binding_digest") != _binding_digest(payload):
        errors.append("EXP-297 reconstruction binding digest mismatch")
    return errors
