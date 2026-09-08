from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.experiments.exp297_challenge_worlds import (
    challenge_contract,
    challenge_contract_digest as current_challenge_contract_digest,
)
from nolane_ai.experiments.exp297_confirmatory_prep import (
    validate_exp297_confirmatory_prep,
)
from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-297-CONFIRMATORY-EXECUTION-AUTH-V1"
EXPERIMENT_ID = "EXP-297"
STATUS = "AUTHORIZED_NOT_EXECUTED"

_VALIDATED_PREP_DIGESTS: set[str] = set()


def _authorization_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def _prep_content_digest(prep: dict[str, Any]) -> str:
    clean = deepcopy(prep)
    clean.pop("prep_digest", None)
    return canonical_sha256(clean)


def _validated_prep(prep: dict[str, Any]) -> list[str]:
    digest = prep.get("prep_digest")
    if not isinstance(digest, str) or not digest or digest != _prep_content_digest(prep):
        return ["EXP-297 prep digest mismatch"]
    if digest not in _VALIDATED_PREP_DIGESTS:
        prep_errors = validate_exp297_confirmatory_prep(prep)
        if prep_errors:
            return ["invalid EXP-297 prep: " + "; ".join(prep_errors)]
        _VALIDATED_PREP_DIGESTS.add(digest)
    return []


def _geometry_from_prep(prep: dict[str, Any]) -> dict[str, int]:
    execution = prep.get("development_execution_artifact") or {}
    config = execution.get("config") or {}
    geometry: dict[str, int] = {}
    for key in ("d_model", "hidden_size", "target_parameters", "max_exact_assignments"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid EXP-297 development geometry: {key}")
        geometry[key] = int(value)
    return geometry


def _expected_reserved(prep: dict[str, Any]) -> tuple[int, list[int]]:
    freeze = prep.get("sample_size_freeze") or {}
    confirmatory_n = freeze.get("confirmatory_n")
    reserved = (prep.get("confirmatory_lineage") or {}).get("reserved_replicate_ids")
    if not isinstance(confirmatory_n, int) or isinstance(confirmatory_n, bool) or not (32 <= confirmatory_n <= 128):
        raise ValueError("EXP-297 prep has no executable confirmatory sample size")
    if not isinstance(reserved, list) or not reserved:
        raise ValueError("EXP-297 prep reserved replicate lineage missing")
    expected = list(range(reserved[0], reserved[0] + len(reserved)))
    if reserved != expected or len(reserved) != confirmatory_n or len(set(reserved)) != len(reserved):
        raise ValueError("EXP-297 prep reserved replicate lineage invalid")
    pilot_ids = (prep.get("pilot_summary") or {}).get("replicate_ids") or []
    if set(reserved) & set(pilot_ids):
        raise ValueError("EXP-297 reserved replicates overlap DEVELOPMENT pilot")
    return confirmatory_n, list(reserved)


def _prep_snapshot(prep: dict[str, Any]) -> dict[str, Any]:
    geometry = _geometry_from_prep(prep)
    confirmatory_n, reserved = _expected_reserved(prep)
    execution = prep.get("development_execution_artifact") or {}
    prep_lineage = prep.get("lineage") or {}
    pair_audit = (execution.get("resource_match") or {}).get("pair_audit") or {}
    pilot_ids = (prep.get("pilot_summary") or {}).get("replicate_ids") or []
    return {
        "prep_digest": prep.get("prep_digest"),
        "prep_status": prep.get("status"),
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "pilot_replicate_ids_digest": canonical_sha256(pilot_ids),
        "frozen_analysis_digest": canonical_sha256(prep.get("frozen_analysis") or {}),
        "sample_size_freeze_digest": canonical_sha256(prep.get("sample_size_freeze") or {}),
        "protocol_digest": execution.get("protocol_digest"),
        "development_execution_digest": execution.get("artifact_digest"),
        "development_code_digest": execution.get("code_digest"),
        "arm_registry_digest": prep_lineage.get("arm_registry_digest"),
        "analysis_code_digest": prep_lineage.get("analysis_code_digest"),
        "pair_audit_digest": canonical_sha256(pair_audit),
        "model_init_seed": execution.get("model_init_seed"),
        "model_geometry": geometry,
    }


def _expected_lineage(
    snapshot: dict[str, Any],
    *,
    evaluator_code_digest: str,
    execution_code_digest: str,
) -> dict[str, Any]:
    return {
        "prep_digest": snapshot.get("prep_digest"),
        "protocol_digest": snapshot.get("protocol_digest"),
        "development_execution_digest": snapshot.get("development_execution_digest"),
        "development_code_digest": snapshot.get("development_code_digest"),
        "arm_registry_digest": snapshot.get("arm_registry_digest"),
        "analysis_code_digest": snapshot.get("analysis_code_digest"),
        "pair_audit_digest": snapshot.get("pair_audit_digest"),
        "model_init_seed": snapshot.get("model_init_seed"),
        "evaluator_code_digest": evaluator_code_digest,
        "execution_code_digest": execution_code_digest,
    }


def _binding_digest(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "prep_snapshot_digest": payload.get("prep_snapshot_digest"),
            "lineage": payload.get("lineage"),
            "model_geometry": payload.get("model_geometry"),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "candidate_count_per_replicate": payload.get("candidate_count_per_replicate"),
            "challenge_contract_digest": payload.get("challenge_contract_digest"),
            "frozen_analysis_digest": payload.get("frozen_analysis_digest"),
            "sample_size_freeze_digest": payload.get("sample_size_freeze_digest"),
            "code_digests": payload.get("code_digests"),
        }
    )


def authorize_exp297_confirmatory_execution(
    *,
    prep_artifact: dict[str, Any],
    challenge_contract_digest: str,
    evaluator_code_digest: str,
    execution_code_digest: str,
) -> dict[str, Any]:
    prep_errors = _validated_prep(prep_artifact)
    if prep_errors:
        raise ValueError("; ".join(prep_errors))
    if prep_artifact.get("status") != "CONFIRMATORY_GATE_A_PREPARED":
        raise ValueError("EXP-297 confirmatory prep is not executable")
    expected_challenge_digest = current_challenge_contract_digest()
    if challenge_contract_digest != expected_challenge_digest:
        raise ValueError("EXP-297 challenge contract digest drift")
    for name, value in (
        ("evaluator_code_digest", evaluator_code_digest),
        ("execution_code_digest", execution_code_digest),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} is required")

    snapshot = _prep_snapshot(prep_artifact)
    contract = challenge_contract()
    if contract.get("candidate_count_per_replicate") != 16:
        raise ValueError("EXP-297 challenge candidate count drift")
    lineage = _expected_lineage(
        snapshot,
        evaluator_code_digest=evaluator_code_digest,
        execution_code_digest=execution_code_digest,
    )
    geometry = deepcopy(snapshot["model_geometry"])
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
        "confirmatory_n": snapshot["confirmatory_n"],
        "reserved_replicate_ids": deepcopy(snapshot["reserved_replicate_ids"]),
        "candidate_count_per_replicate": 16,
        "challenge_contract_digest": expected_challenge_digest,
        "model_geometry": geometry,
        "court_ceiling": geometry["max_exact_assignments"],
        "frozen_analysis_digest": snapshot["frozen_analysis_digest"],
        "sample_size_freeze_digest": snapshot["sample_size_freeze_digest"],
        "code_digests": {
            "evaluator": evaluator_code_digest,
            "execution": execution_code_digest,
        },
        "lineage": lineage,
        "prep_snapshot": snapshot,
        "prep_snapshot_digest": canonical_sha256(snapshot),
        "binding_digest": "",
        "authorization_digest": "",
    }
    payload["binding_digest"] = _binding_digest(payload)
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp297_confirmatory_execution_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-297 execution authorization: " + "; ".join(errors))
    return payload


def validate_exp297_confirmatory_execution_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-297 execution authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-297 execution authorization cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-297 execution authorization status drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
        "semantic_authority_promoted",
    ):
        if payload.get(flag) is not False:
            errors.append(f"EXP-297 execution authorization forbidden flag enabled: {flag}")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-297 execution authorization cannot materialize seeds")
    rendered = repr(payload).lower()
    for forbidden in ("beacon_receipt", "challenge_seed", "challenge_candidates"):
        if forbidden in rendered:
            errors.append(f"EXP-297 execution authorization contains forbidden pre-freeze material: {forbidden}")
    if payload.get("authorization_digest") != _authorization_digest(payload):
        errors.append("EXP-297 execution authorization digest mismatch")

    snapshot = payload.get("prep_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("EXP-297 execution authorization prep snapshot missing")
        return errors
    if payload.get("prep_snapshot_digest") != canonical_sha256(snapshot):
        errors.append("EXP-297 execution authorization prep snapshot digest mismatch")
    if snapshot.get("prep_status") != "CONFIRMATORY_GATE_A_PREPARED":
        errors.append("EXP-297 execution authorization prep snapshot not executable")

    geometry = snapshot.get("model_geometry") or {}
    if payload.get("model_geometry") != geometry:
        errors.append("EXP-297 execution authorization model geometry mismatch")
    if payload.get("court_ceiling") != geometry.get("max_exact_assignments"):
        errors.append("EXP-297 execution authorization court ceiling mismatch")
    if payload.get("confirmatory_n") != snapshot.get("confirmatory_n"):
        errors.append("EXP-297 execution authorization confirmatory n mismatch")
    if payload.get("reserved_replicate_ids") != snapshot.get("reserved_replicate_ids"):
        errors.append("EXP-297 execution authorization reserved replicate lineage mismatch")

    contract = challenge_contract()
    expected_challenge_digest = current_challenge_contract_digest()
    if payload.get("challenge_contract_digest") != expected_challenge_digest:
        errors.append("EXP-297 execution authorization challenge contract mismatch")
    if payload.get("candidate_count_per_replicate") != contract.get("candidate_count_per_replicate"):
        errors.append("EXP-297 execution authorization candidate count mismatch")
    if payload.get("frozen_analysis_digest") != snapshot.get("frozen_analysis_digest"):
        errors.append("EXP-297 execution authorization frozen-analysis digest mismatch")
    if payload.get("sample_size_freeze_digest") != snapshot.get("sample_size_freeze_digest"):
        errors.append("EXP-297 execution authorization sample-size digest mismatch")

    code_digests = payload.get("code_digests") or {}
    evaluator_code_digest = code_digests.get("evaluator")
    execution_code_digest = code_digests.get("execution")
    if not isinstance(evaluator_code_digest, str) or not evaluator_code_digest:
        errors.append("EXP-297 evaluator code digest missing")
    if not isinstance(execution_code_digest, str) or not execution_code_digest:
        errors.append("EXP-297 execution code digest missing")
    if errors:
        return errors

    expected_lineage = _expected_lineage(
        snapshot,
        evaluator_code_digest=evaluator_code_digest,
        execution_code_digest=execution_code_digest,
    )
    if payload.get("lineage") != expected_lineage:
        errors.append("EXP-297 execution authorization lineage mismatch")
    if payload.get("binding_digest") != _binding_digest(payload):
        errors.append("EXP-297 execution authorization binding digest mismatch")
    return errors
