from __future__ import annotations

from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp282_confirmatory_execution_court import validate_exp282_confirmatory_execution_authorization
from .exp282_paired_runner import validate_exp282_paired_development

SCHEMA = "NLM-EXP-282-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"


def _reconstruction_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("reconstruction_digest", None)
    return canonical_sha256(clean)


def validate_exp282_confirmatory_reconstruction(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 reconstruction authorization schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("reconstruction authorization cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("reconstruction authorization cannot promote a neural claim")
    if payload.get("status") != "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED":
        errors.append("reconstruction authorization status drift")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("reconstruction authorization cannot consume confirmatory data")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("reconstruction authorization cannot materialize confirmatory seeds")
    if "seeds" in payload:
        errors.append("reconstruction authorization cannot contain materialized seeds")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(confirmatory_n, int) or not (32 <= confirmatory_n <= 128):
        errors.append("reconstruction confirmatory n is outside frozen bounds")
    elif len(reserved) != confirmatory_n:
        errors.append("reconstruction reserved replicate count mismatch")
    if len(set(reserved)) != len(reserved):
        errors.append("reconstruction reserved replicate IDs must be unique")
    if reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("reconstruction reserved replicate lineage must be contiguous and ordered")

    checkpoint_sha = payload.get("checkpoint_sha256")
    if not checkpoint_sha:
        errors.append("reconstruction authorization checkpoint identity is missing")
    contract = payload.get("execution_contract") or {}
    contract_digest = payload.get("execution_contract_digest")
    if not contract_digest or contract_digest != canonical_sha256(contract):
        errors.append("execution contract digest mismatch")
    if not contract.get("root_seed"):
        errors.append("execution contract root seed is missing")
    arm = contract.get("arm_geometry") or {}
    world = contract.get("world_geometry") or {}
    for key in ("d_model", "hidden_size", "target_parameters"):
        if int(arm.get(key, 0) or 0) <= 0:
            errors.append(f"execution arm geometry {key} is invalid")
    for key in ("batch_size", "timesteps", "variables", "d_model"):
        if int(world.get(key, 0) or 0) <= 0:
            errors.append(f"execution world geometry {key} is invalid")
    if arm.get("d_model") != world.get("d_model"):
        errors.append("execution contract d_model mismatch")

    lineage = payload.get("lineage") or {}
    for key in (
        "protocol_digest",
        "paired_execution_artifact_digest",
        "execution_authorization_digest",
        "paired_checkpoint_sha256",
    ):
        if not lineage.get(key):
            errors.append(f"missing reconstruction lineage {key}")
    if checkpoint_sha and lineage.get("paired_checkpoint_sha256") != checkpoint_sha:
        errors.append("reconstruction checkpoint lineage mismatch")

    preflight = payload.get("preflight") or {}
    required = (
        "paired_artifact_valid",
        "execution_authorization_valid",
        "checkpoint_identity_match",
        "execution_contract_digest_match",
        "geometry_closed",
        "confirmatory_data_unconsumed",
        "seeds_unmaterialized",
        "reserved_lineage_bound",
    )
    if any(preflight.get(name) is not True for name in required):
        errors.append("reconstruction preflight is incomplete")
    if preflight.get("all_checks_passed") is not True:
        errors.append("reconstruction preflight did not pass")

    if payload.get("reconstruction_digest") not in (None, ""):
        if payload.get("reconstruction_digest") != _reconstruction_digest(payload):
            errors.append("reconstruction authorization digest mismatch")
    return errors


def authorize_exp282_confirmatory_reconstruction(
    *,
    execution_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
) -> dict[str, Any]:
    execution_errors = validate_exp282_paired_development(execution_artifact)
    if execution_errors:
        raise ValueError("invalid paired development artifact: " + "; ".join(execution_errors))
    authorization_errors = validate_exp282_confirmatory_execution_authorization(execution_authorization)
    if authorization_errors:
        raise ValueError("invalid execution authorization: " + "; ".join(authorization_errors))

    checkpoint = execution_artifact.get("checkpoint") or {}
    if checkpoint.get("schema") != "NLM-EXP-282-PAIRED-CHECKPOINT-V1":
        raise ValueError("paired checkpoint is missing or invalid")
    checkpoint_sha = checkpoint.get("checkpoint_sha256")
    auth_checkpoint_sha = (execution_authorization.get("lineage") or {}).get("paired_checkpoint_sha256")
    if not checkpoint_sha or checkpoint_sha != auth_checkpoint_sha:
        raise ValueError("checkpoint identity mismatch between paired execution and authorization")

    contract = checkpoint.get("execution_contract") or {}
    contract_digest = checkpoint.get("execution_contract_digest")
    if not contract_digest or contract_digest != canonical_sha256(contract):
        raise ValueError("execution contract digest mismatch")
    if contract.get("root_seed") != execution_artifact.get("root_seed"):
        raise ValueError("execution contract root-seed mismatch")
    if contract.get("arm_geometry") != execution_artifact.get("arm_geometry"):
        raise ValueError("execution contract arm-geometry mismatch")
    if contract.get("world_geometry") != execution_artifact.get("world_geometry"):
        raise ValueError("execution contract world-geometry mismatch")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED",
        "scope": "exp282-confirmatory-reconstruction-authorization-only",
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "confirmatory_n": int(execution_authorization["confirmatory_n"]),
        "reserved_replicate_ids": list(execution_authorization["reserved_replicate_ids"]),
        "frozen_analysis_digest": execution_authorization.get("frozen_analysis_digest"),
        "sample_size_freeze_digest": execution_authorization.get("sample_size_freeze_digest"),
        "checkpoint_sha256": checkpoint_sha,
        "execution_contract": contract,
        "execution_contract_digest": contract_digest,
        "model_init_seed": execution_artifact.get("model_init_seed"),
        "lineage": {
            "protocol_digest": execution_artifact.get("protocol_digest"),
            "paired_execution_artifact_digest": execution_artifact.get("artifact_digest"),
            "execution_authorization_digest": execution_authorization.get("authorization_digest"),
            "paired_checkpoint_sha256": checkpoint_sha,
        },
        "preflight": {
            "paired_artifact_valid": True,
            "execution_authorization_valid": True,
            "checkpoint_identity_match": checkpoint_sha == auth_checkpoint_sha,
            "execution_contract_digest_match": contract_digest == canonical_sha256(contract),
            "geometry_closed": bool(contract.get("arm_geometry") and contract.get("world_geometry")),
            "confirmatory_data_unconsumed": execution_authorization.get("confirmatory_data_consumed") is False,
            "seeds_unmaterialized": execution_authorization.get("seed_materialization_status") == "NOT_EXECUTED",
            "reserved_lineage_bound": len(execution_authorization.get("reserved_replicate_ids") or []) == int(execution_authorization.get("confirmatory_n", 0) or 0),
            "all_checks_passed": True,
        },
        "remaining_blockers": [
            "confirmatory observations have not been executed",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "reconstruction_digest": "",
    }
    payload["reconstruction_digest"] = _reconstruction_digest(payload)
    errors = validate_exp282_confirmatory_reconstruction(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 reconstruction authorization: " + "; ".join(errors))
    return payload
