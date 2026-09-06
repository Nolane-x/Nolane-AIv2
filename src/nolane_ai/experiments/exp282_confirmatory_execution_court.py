from __future__ import annotations

from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp282_confirmatory_prep import validate_exp282_confirmatory_prep

SCHEMA = "NLM-EXP-282-CONFIRMATORY-EXECUTION-AUTH-V1"


def _authorization_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("authorization_digest", None)
    return canonical_sha256(clean)


def validate_exp282_confirmatory_execution_authorization(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("invalid EXP-282 execution authorization schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("execution authorization cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("execution authorization cannot promote a neural claim")
    if payload.get("status") != "AUTHORIZED_NOT_EXECUTED":
        errors.append("execution authorization status drift")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("execution authorization cannot consume confirmatory data")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("execution authorization cannot materialize confirmatory seeds")
    if "seeds" in payload:
        errors.append("execution authorization cannot contain materialized seeds")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(confirmatory_n, int) or not (32 <= confirmatory_n <= 128):
        errors.append("execution authorization confirmatory n is outside frozen bounds")
    elif len(reserved) != confirmatory_n:
        errors.append("execution authorization reserved replicate count mismatch")
    if len(set(reserved)) != len(reserved):
        errors.append("execution authorization reserved replicate IDs must be unique")
    if reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("execution authorization reserved replicate lineage must be contiguous and ordered")

    lineage = payload.get("lineage") or {}
    for key in ("prep_digest", "protocol_digest", "execution_code_digest", "analysis_code_digest", "paired_checkpoint_sha256"):
        if not lineage.get(key):
            errors.append(f"missing execution authorization lineage {key}")

    preflight = payload.get("preflight") or {}
    required_checks = (
        "prep_semantically_valid",
        "prep_status_prepared",
        "confirmatory_ids_reserved",
        "development_lineages_disjoint",
        "confirmatory_data_unconsumed",
        "seeds_unmaterialized",
    )
    if any(preflight.get(name) is not True for name in required_checks):
        errors.append("execution authorization preflight is incomplete")
    if preflight.get("all_checks_passed") is not True:
        errors.append("execution authorization preflight did not pass")

    if payload.get("authorization_digest") not in (None, ""):
        if payload.get("authorization_digest") != _authorization_digest(payload):
            errors.append("execution authorization digest mismatch")
    return errors


def authorize_exp282_confirmatory_execution(
    *,
    prep_artifact: dict[str, Any],
    execution_code_digest: str,
) -> dict[str, Any]:
    if not execution_code_digest:
        raise ValueError("execution_code_digest is required")

    prep_errors = validate_exp282_confirmatory_prep(prep_artifact)
    if prep_errors:
        raise ValueError("invalid confirmatory prep: " + "; ".join(prep_errors))
    if prep_artifact.get("status") != "CONFIRMATORY_OPEN_PREPARED":
        raise ValueError("confirmatory prep is not CONFIRMATORY_OPEN_PREPARED")

    freeze = prep_artifact["sample_size_freeze"]
    confirmatory_lineage = prep_artifact["confirmatory_lineage"]
    pilot = prep_artifact["pilot_summary"]
    reserved = list(confirmatory_lineage["reserved_replicate_ids"])
    confirmatory_n = int(freeze["confirmatory_n"])
    development_ids = set(pilot["replicate_ids"]) | set(pilot["training_replicate_ids"])
    if development_ids.intersection(reserved):
        raise ValueError("reserved confirmatory IDs overlap development lineage")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "AUTHORIZED_NOT_EXECUTED",
        "scope": "exp282-confirmatory-open-execution-authorization-only",
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "frozen_analysis_digest": canonical_sha256(prep_artifact["frozen_analysis"]),
        "sample_size_freeze_digest": canonical_sha256(prep_artifact["sample_size_freeze"]),
        "lineage": {
            "prep_digest": prep_artifact["prep_digest"],
            "protocol_digest": prep_artifact["lineage"]["protocol_digest"],
            "analysis_code_digest": prep_artifact["lineage"]["analysis_code_digest"],
            "execution_code_digest": execution_code_digest,
            "paired_checkpoint_sha256": prep_artifact["lineage"]["paired_checkpoint_sha256"],
        },
        "preflight": {
            "prep_semantically_valid": True,
            "prep_status_prepared": True,
            "confirmatory_ids_reserved": len(reserved) == confirmatory_n,
            "development_lineages_disjoint": not bool(development_ids.intersection(reserved)),
            "confirmatory_data_unconsumed": prep_artifact["confirmatory_data_consumed"] is False,
            "seeds_unmaterialized": confirmatory_lineage["seed_materialization_status"] == "NOT_EXECUTED",
            "all_checks_passed": True,
        },
        "remaining_blockers": [
            "confirmatory observations have not been executed",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "authorization_digest": "",
    }
    payload["authorization_digest"] = _authorization_digest(payload)
    errors = validate_exp282_confirmatory_execution_authorization(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 execution authorization: " + "; ".join(errors))
    return payload
