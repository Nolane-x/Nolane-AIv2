from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256, require_canonical_stage_a_v1_digest
from .exp282_confirmatory_analysis import (
    build_exp282_confirmatory_analysis,
    validate_exp282_confirmatory_analysis,
)
from .exp282_confirmatory_execution_court import validate_exp282_confirmatory_execution_authorization
from .exp282_confirmatory_executor import (
    execute_exp282_confirmatory_open,
    validate_exp282_confirmatory_open_raw,
)
from .exp282_confirmatory_prep import validate_exp282_confirmatory_prep
from .exp282_paired_runner import validate_exp282_paired_development
from .exp282_reconstruction_court import validate_exp282_confirmatory_reconstruction

SEAL_SCHEMA = "NLM-EXP-282-CONFIRMATORY-CEREMONY-SEAL-V1"
RESULT_SCHEMA = "NLM-EXP-282-CONFIRMATORY-CEREMONY-RESULT-V1"


def _seal_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("ceremony_seal_digest", None)
    return canonical_sha256(clean)


def _result_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("ceremony_result_digest", None)
    return canonical_sha256(clean)


def _deterministic_reserved_ids(prep: dict[str, Any]) -> list[int]:
    freeze = prep.get("sample_size_freeze") or {}
    confirmatory_n = freeze.get("confirmatory_n")
    if not isinstance(confirmatory_n, int) or confirmatory_n <= 0:
        return []
    pilot = prep.get("pilot_summary") or {}
    pilot_ids = [int(item) for item in (pilot.get("replicate_ids") or [])]
    training_ids = [int(item) for item in (pilot.get("training_replicate_ids") or [])]
    if not pilot_ids or not training_ids:
        return []
    start = max([*pilot_ids, *training_ids]) + 1
    return list(range(start, start + confirmatory_n))


def validate_exp282_confirmatory_ceremony_seal(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SEAL_SCHEMA:
        errors.append("invalid EXP-282 ceremony seal schema")
    if payload.get("evidence_level") != "EV-E2":
        errors.append("ceremony seal cannot claim EV-E3+")
    if payload.get("decision") != "UNVERIFIED":
        errors.append("ceremony seal cannot promote a neural claim")
    if payload.get("status") != "CEREMONY_SEALED_NOT_EXECUTED":
        errors.append("ceremony seal status drift")
    if payload.get("confirmatory_data_consumed") is not False:
        errors.append("ceremony seal cannot consume confirmatory data")
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("ceremony seal cannot materialize confirmatory seeds")
    if payload.get("challenge_materialized") is not False:
        errors.append("ceremony seal cannot materialize challenge randomness")
    if "seeds" in payload:
        errors.append("ceremony seal cannot contain materialized seeds")

    confirmatory_n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(confirmatory_n, int) or not (32 <= confirmatory_n <= 128):
        errors.append("ceremony confirmatory n is outside frozen bounds")
    elif len(reserved) != confirmatory_n:
        errors.append("ceremony reserved replicate count mismatch")
    if len(set(reserved)) != len(reserved):
        errors.append("ceremony reserved replicate IDs must be unique")
    if reserved and reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("ceremony reserved replicate IDs must be contiguous and ordered")

    lineage = payload.get("lineage") or {}
    required_lineage = (
        "protocol_digest",
        "paired_execution_artifact_digest",
        "prep_digest",
        "execution_authorization_digest",
        "reconstruction_digest",
        "paired_checkpoint_sha256",
        "execution_contract_digest",
        "frozen_analysis_digest",
        "sample_size_freeze_digest",
        "analysis_code_digest",
        "execution_code_digest",
        "ceremony_code_digest",
    )
    for key in required_lineage:
        if not lineage.get(key):
            errors.append(f"missing ceremony seal lineage {key}")
    try:
        require_canonical_stage_a_v1_digest(str(lineage.get("protocol_digest") or ""))
    except ValueError:
        errors.append("ceremony seal protocol digest is not canonical frozen V1")
    code_digests = [lineage.get(key) for key in ("analysis_code_digest", "execution_code_digest", "ceremony_code_digest")]
    if any(not value for value in code_digests) or len(set(code_digests)) != 1:
        errors.append("ceremony code digest freeze mismatch")

    authorities = payload.get("authorities") or {}
    prep = authorities.get("prep") or {}
    authorization = authorities.get("execution_authorization") or {}
    reconstruction = authorities.get("reconstruction_authorization") or {}
    prep_errors = validate_exp282_confirmatory_prep(prep) if isinstance(prep, dict) else ["prep is not an object"]
    auth_errors = (
        validate_exp282_confirmatory_execution_authorization(authorization)
        if isinstance(authorization, dict)
        else ["execution authorization is not an object"]
    )
    reconstruction_errors = (
        validate_exp282_confirmatory_reconstruction(reconstruction)
        if isinstance(reconstruction, dict)
        else ["reconstruction authorization is not an object"]
    )
    if prep_errors:
        errors.append("embedded ceremony prep is invalid: " + "; ".join(prep_errors))
    if auth_errors:
        errors.append("embedded execution authorization is invalid: " + "; ".join(auth_errors))
    if reconstruction_errors:
        errors.append("embedded reconstruction authorization is invalid: " + "; ".join(reconstruction_errors))

    if not prep_errors:
        expected_reserved = _deterministic_reserved_ids(prep)
        prep_reserved = list((prep.get("confirmatory_lineage") or {}).get("reserved_replicate_ids") or [])
        prep_n = (prep.get("sample_size_freeze") or {}).get("confirmatory_n")
        if prep.get("status") != "CONFIRMATORY_OPEN_PREPARED":
            errors.append("ceremony seal requires CONFIRMATORY_OPEN_PREPARED prep")
        if prep_n != confirmatory_n or prep_reserved != reserved:
            errors.append("ceremony seal/prep confirmatory lineage mismatch")
        if expected_reserved != reserved:
            errors.append("ceremony reserved replicate lineage violates deterministic reservation rule")
        if prep.get("prep_digest") != lineage.get("prep_digest"):
            errors.append("ceremony prep digest lineage mismatch")
        if (prep.get("lineage") or {}).get("protocol_digest") != lineage.get("protocol_digest"):
            errors.append("ceremony prep protocol lineage mismatch")
        if (prep.get("lineage") or {}).get("analysis_code_digest") != lineage.get("analysis_code_digest"):
            errors.append("ceremony prep analysis-code lineage mismatch")
        if canonical_sha256(prep.get("frozen_analysis") or {}) != lineage.get("frozen_analysis_digest"):
            errors.append("ceremony frozen-analysis digest mismatch")
        if canonical_sha256(prep.get("sample_size_freeze") or {}) != lineage.get("sample_size_freeze_digest"):
            errors.append("ceremony sample-size-freeze digest mismatch")

    if not auth_errors:
        auth_lineage = authorization.get("lineage") or {}
        if authorization.get("confirmatory_n") != confirmatory_n or authorization.get("reserved_replicate_ids") != reserved:
            errors.append("ceremony execution-authorization confirmatory lineage mismatch")
        if authorization.get("authorization_digest") != lineage.get("execution_authorization_digest"):
            errors.append("ceremony execution-authorization digest mismatch")
        if auth_lineage.get("prep_digest") != lineage.get("prep_digest"):
            errors.append("ceremony execution-authorization prep lineage mismatch")
        if auth_lineage.get("execution_code_digest") != lineage.get("execution_code_digest"):
            errors.append("ceremony execution-code lineage mismatch")
        if auth_lineage.get("analysis_code_digest") != lineage.get("analysis_code_digest"):
            errors.append("ceremony analysis-code lineage mismatch")
        if auth_lineage.get("paired_checkpoint_sha256") != lineage.get("paired_checkpoint_sha256"):
            errors.append("ceremony execution-authorization checkpoint lineage mismatch")

    if not reconstruction_errors:
        reconstruction_lineage = reconstruction.get("lineage") or {}
        if reconstruction.get("confirmatory_n") != confirmatory_n or reconstruction.get("reserved_replicate_ids") != reserved:
            errors.append("ceremony reconstruction confirmatory lineage mismatch")
        if reconstruction.get("reconstruction_digest") != lineage.get("reconstruction_digest"):
            errors.append("ceremony reconstruction digest mismatch")
        if reconstruction.get("checkpoint_sha256") != lineage.get("paired_checkpoint_sha256"):
            errors.append("ceremony reconstruction checkpoint lineage mismatch")
        if reconstruction.get("execution_contract_digest") != lineage.get("execution_contract_digest"):
            errors.append("ceremony reconstruction execution-contract lineage mismatch")
        if reconstruction_lineage.get("paired_execution_artifact_digest") != lineage.get("paired_execution_artifact_digest"):
            errors.append("ceremony reconstruction paired-execution lineage mismatch")
        if reconstruction_lineage.get("execution_authorization_digest") != lineage.get("execution_authorization_digest"):
            errors.append("ceremony reconstruction execution-authorization lineage mismatch")

    preflight = payload.get("preflight") or {}
    required_checks = (
        "prep_valid",
        "execution_authorization_valid",
        "reconstruction_authorization_valid",
        "confirmatory_lineage_closed",
        "checkpoint_identity_closed",
        "code_tree_frozen",
        "confirmatory_data_unconsumed",
        "seeds_unmaterialized",
        "challenge_unmaterialized",
    )
    if any(preflight.get(name) is not True for name in required_checks):
        errors.append("ceremony seal preflight is incomplete")
    if preflight.get("all_checks_passed") is not True:
        errors.append("ceremony seal preflight did not pass")

    if payload.get("ceremony_seal_digest") not in (None, "") and payload.get("ceremony_seal_digest") != _seal_digest(payload):
        errors.append("ceremony seal digest mismatch")
    return errors


def seal_exp282_confirmatory_ceremony(
    *,
    protocol_digest: str,
    paired_execution_artifact: dict[str, Any],
    prep_artifact: dict[str, Any],
    execution_authorization: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    ceremony_code_digest: str,
) -> dict[str, Any]:
    require_canonical_stage_a_v1_digest(protocol_digest)
    if not ceremony_code_digest:
        raise ValueError("ceremony_code_digest is required")

    validators = (
        ("paired execution artifact", validate_exp282_paired_development(paired_execution_artifact)),
        ("confirmatory prep", validate_exp282_confirmatory_prep(prep_artifact)),
        ("execution authorization", validate_exp282_confirmatory_execution_authorization(execution_authorization)),
        ("reconstruction authorization", validate_exp282_confirmatory_reconstruction(reconstruction_authorization)),
    )
    for label, errors in validators:
        if errors:
            raise ValueError(f"invalid {label}: " + "; ".join(errors))
    if prep_artifact.get("status") != "CONFIRMATORY_OPEN_PREPARED":
        raise ValueError("confirmatory prep is not CONFIRMATORY_OPEN_PREPARED")

    execution_digest = paired_execution_artifact.get("artifact_digest")
    prep_lineage = prep_artifact.get("lineage") or {}
    auth_lineage = execution_authorization.get("lineage") or {}
    reconstruction_lineage = reconstruction_authorization.get("lineage") or {}
    if any(
        value != protocol_digest
        for value in (
            paired_execution_artifact.get("protocol_digest"),
            prep_lineage.get("protocol_digest"),
            auth_lineage.get("protocol_digest"),
            reconstruction_lineage.get("protocol_digest"),
        )
    ):
        raise ValueError("ceremony protocol lineage mismatch")
    if prep_lineage.get("execution_artifact_digest") != execution_digest:
        raise ValueError("ceremony prep/paired execution digest mismatch")
    if reconstruction_lineage.get("paired_execution_artifact_digest") != execution_digest:
        raise ValueError("ceremony reconstruction/paired execution digest mismatch")

    analysis_code_digest = prep_lineage.get("analysis_code_digest")
    execution_code_digest = auth_lineage.get("execution_code_digest")
    if analysis_code_digest != ceremony_code_digest or execution_code_digest != ceremony_code_digest:
        raise ValueError("ceremony code digest must match frozen analysis and execution code digests")
    if auth_lineage.get("analysis_code_digest") != ceremony_code_digest:
        raise ValueError("ceremony analysis code digest mismatch")

    confirmatory_n = int((prep_artifact.get("sample_size_freeze") or {}).get("confirmatory_n", 0) or 0)
    reserved = list((prep_artifact.get("confirmatory_lineage") or {}).get("reserved_replicate_ids") or [])
    expected_reserved = _deterministic_reserved_ids(prep_artifact)
    if expected_reserved != reserved:
        raise ValueError("ceremony prep reserved IDs violate deterministic reservation rule")
    if execution_authorization.get("confirmatory_n") != confirmatory_n or execution_authorization.get("reserved_replicate_ids") != reserved:
        raise ValueError("ceremony execution authorization lineage mismatch")
    if reconstruction_authorization.get("confirmatory_n") != confirmatory_n or reconstruction_authorization.get("reserved_replicate_ids") != reserved:
        raise ValueError("ceremony reconstruction lineage mismatch")

    checkpoint_sha = reconstruction_authorization.get("checkpoint_sha256")
    execution_checkpoint = paired_execution_artifact.get("checkpoint") or {}
    if not checkpoint_sha or checkpoint_sha != execution_checkpoint.get("checkpoint_sha256"):
        raise ValueError("ceremony checkpoint identity mismatch")
    execution_contract_digest = reconstruction_authorization.get("execution_contract_digest")
    if not execution_contract_digest or execution_contract_digest != execution_checkpoint.get("execution_contract_digest"):
        raise ValueError("ceremony execution contract identity mismatch")

    frozen_analysis_digest = canonical_sha256(prep_artifact.get("frozen_analysis") or {})
    sample_size_freeze_digest = canonical_sha256(prep_artifact.get("sample_size_freeze") or {})
    if execution_authorization.get("frozen_analysis_digest") != frozen_analysis_digest:
        raise ValueError("ceremony execution authorization frozen-analysis mismatch")
    if reconstruction_authorization.get("frozen_analysis_digest") != frozen_analysis_digest:
        raise ValueError("ceremony reconstruction frozen-analysis mismatch")
    if execution_authorization.get("sample_size_freeze_digest") != sample_size_freeze_digest:
        raise ValueError("ceremony execution authorization sample-size freeze mismatch")
    if reconstruction_authorization.get("sample_size_freeze_digest") != sample_size_freeze_digest:
        raise ValueError("ceremony reconstruction sample-size freeze mismatch")

    payload: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": "CEREMONY_SEALED_NOT_EXECUTED",
        "scope": "exp282-confirmatory-open-two-phase-seal",
        "confirmatory_data_consumed": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "challenge_materialized": False,
        "confirmatory_n": confirmatory_n,
        "reserved_replicate_ids": reserved,
        "authorities": {
            "prep": deepcopy(prep_artifact),
            "execution_authorization": deepcopy(execution_authorization),
            "reconstruction_authorization": deepcopy(reconstruction_authorization),
        },
        "lineage": {
            "protocol_digest": protocol_digest,
            "paired_execution_artifact_digest": execution_digest,
            "prep_digest": prep_artifact.get("prep_digest"),
            "execution_authorization_digest": execution_authorization.get("authorization_digest"),
            "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
            "paired_checkpoint_sha256": checkpoint_sha,
            "execution_contract_digest": execution_contract_digest,
            "frozen_analysis_digest": frozen_analysis_digest,
            "sample_size_freeze_digest": sample_size_freeze_digest,
            "analysis_code_digest": analysis_code_digest,
            "execution_code_digest": execution_code_digest,
            "ceremony_code_digest": ceremony_code_digest,
        },
        "preflight": {
            "prep_valid": True,
            "execution_authorization_valid": True,
            "reconstruction_authorization_valid": True,
            "confirmatory_lineage_closed": expected_reserved == reserved,
            "checkpoint_identity_closed": checkpoint_sha == execution_checkpoint.get("checkpoint_sha256"),
            "code_tree_frozen": analysis_code_digest == execution_code_digest == ceremony_code_digest,
            "confirmatory_data_unconsumed": True,
            "seeds_unmaterialized": True,
            "challenge_unmaterialized": True,
            "all_checks_passed": True,
        },
        "remaining_blockers": [
            "confirmatory-open observations have not been executed",
            "secure cross-host artifact transport / replay lock is not implemented",
            "post-freeze challenge beacon and independent replication remain open",
        ],
        "ceremony_seal_digest": "",
    }
    payload["ceremony_seal_digest"] = _seal_digest(payload)
    errors = validate_exp282_confirmatory_ceremony_seal(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 ceremony seal: " + "; ".join(errors))
    return payload


def validate_exp282_confirmatory_ceremony_result(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != RESULT_SCHEMA:
        errors.append("invalid EXP-282 ceremony result schema")
    if payload.get("status") != "CEREMONY_EXECUTED_AND_ANALYZED":
        errors.append("ceremony result status drift")
    if payload.get("confirmatory_data_consumed") is not True:
        errors.append("ceremony result must record consumed confirmatory data")
    if payload.get("challenge_materialized") is not False:
        errors.append("ceremony result cannot materialize challenge randomness")

    artifacts = payload.get("artifacts") or {}
    seal = artifacts.get("seal") or {}
    raw = artifacts.get("raw") or {}
    analysis = artifacts.get("analysis") or {}
    seal_errors = validate_exp282_confirmatory_ceremony_seal(seal) if isinstance(seal, dict) else ["seal is not an object"]
    raw_errors = validate_exp282_confirmatory_open_raw(raw) if isinstance(raw, dict) else ["raw is not an object"]
    analysis_errors = validate_exp282_confirmatory_analysis(analysis) if isinstance(analysis, dict) else ["analysis is not an object"]
    if seal_errors:
        errors.append("embedded ceremony seal is invalid: " + "; ".join(seal_errors))
    if raw_errors:
        errors.append("embedded raw artifact is invalid: " + "; ".join(raw_errors))
    if analysis_errors:
        errors.append("embedded analysis artifact is invalid: " + "; ".join(analysis_errors))

    lineage = payload.get("lineage") or {}
    if not seal_errors and seal.get("ceremony_seal_digest") != lineage.get("ceremony_seal_digest"):
        errors.append("ceremony result seal digest lineage mismatch")
    if not raw_errors and raw.get("artifact_digest") != lineage.get("raw_artifact_digest"):
        errors.append("ceremony result raw digest lineage mismatch")
    if not analysis_errors and analysis.get("analysis_digest") != lineage.get("analysis_digest"):
        errors.append("ceremony result analysis digest lineage mismatch")

    if not seal_errors and not raw_errors:
        if raw.get("confirmatory_n") != seal.get("confirmatory_n"):
            errors.append("ceremony result raw confirmatory n mismatch")
        if raw.get("reserved_replicate_ids") != seal.get("reserved_replicate_ids"):
            errors.append("ceremony result raw reserved lineage mismatch")
        raw_lineage = raw.get("lineage") or {}
        seal_lineage = seal.get("lineage") or {}
        for key in ("protocol_digest", "paired_execution_artifact_digest", "reconstruction_digest", "paired_checkpoint_sha256", "execution_contract_digest"):
            if raw_lineage.get(key) != seal_lineage.get(key):
                errors.append(f"ceremony result raw/seal {key} mismatch")
    if not seal_errors and not analysis_errors:
        if analysis.get("confirmatory_n") != seal.get("confirmatory_n"):
            errors.append("ceremony result analysis confirmatory n mismatch")
        if analysis.get("reserved_replicate_ids") != seal.get("reserved_replicate_ids"):
            errors.append("ceremony result analysis reserved lineage mismatch")
        if analysis.get("decision") != payload.get("decision"):
            errors.append("ceremony result decision does not match Analysis Court")
        if analysis.get("evidence_level") != payload.get("evidence_level"):
            errors.append("ceremony result evidence level does not match Analysis Court")

    if payload.get("ceremony_result_digest") not in (None, "") and payload.get("ceremony_result_digest") != _result_digest(payload):
        errors.append("ceremony result digest mismatch")
    return errors


def execute_exp282_confirmatory_ceremony(
    *,
    protocol: dict[str, Any],
    protocol_digest: str,
    ceremony_seal: dict[str, Any],
    paired_execution_artifact: dict[str, Any],
    checkpoint_path: str | Path,
    ceremony_code_digest: str,
) -> dict[str, Any]:
    require_canonical_stage_a_v1_digest(protocol_digest)
    seal_errors = validate_exp282_confirmatory_ceremony_seal(ceremony_seal)
    if seal_errors:
        raise ValueError("invalid confirmatory ceremony seal: " + "; ".join(seal_errors))
    seal_lineage = ceremony_seal.get("lineage") or {}
    if ceremony_code_digest != seal_lineage.get("ceremony_code_digest"):
        raise ValueError("current ceremony code digest does not match sealed source tree")
    if protocol_digest != seal_lineage.get("protocol_digest"):
        raise ValueError("ceremony protocol digest does not match seal")

    paired_errors = validate_exp282_paired_development(paired_execution_artifact)
    if paired_errors:
        raise ValueError("invalid paired execution artifact: " + "; ".join(paired_errors))
    if paired_execution_artifact.get("artifact_digest") != seal_lineage.get("paired_execution_artifact_digest"):
        raise ValueError("paired execution artifact digest does not match ceremony seal")

    checkpoint_path = Path(checkpoint_path)
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if actual_checkpoint_sha != seal_lineage.get("paired_checkpoint_sha256"):
        raise ValueError("ceremony checkpoint SHA mismatch")

    authorities = ceremony_seal["authorities"]
    prep = deepcopy(authorities["prep"])
    reconstruction = deepcopy(authorities["reconstruction_authorization"])
    raw = execute_exp282_confirmatory_open(
        protocol=protocol,
        protocol_digest=protocol_digest,
        paired_execution_artifact=paired_execution_artifact,
        reconstruction_authorization=reconstruction,
        checkpoint_path=checkpoint_path,
        executor_code_digest=ceremony_code_digest,
    )
    raw_errors = validate_exp282_confirmatory_open_raw(raw)
    if raw_errors:
        raise RuntimeError("invalid raw artifact returned by sealed executor: " + "; ".join(raw_errors))
    if raw.get("challenge_materialized") is not False:
        raise RuntimeError("sealed executor materialized challenge randomness")

    analysis = build_exp282_confirmatory_analysis(
        protocol=protocol,
        protocol_digest=protocol_digest,
        prep_artifact=prep,
        paired_execution_artifact=paired_execution_artifact,
        reconstruction_authorization=reconstruction,
        raw_artifact=raw,
        analysis_code_digest=ceremony_code_digest,
    )
    analysis_errors = validate_exp282_confirmatory_analysis(analysis)
    if analysis_errors:
        raise RuntimeError("invalid Analysis Court artifact: " + "; ".join(analysis_errors))

    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "evidence_level": analysis["evidence_level"],
        "decision": analysis["decision"],
        "status": "CEREMONY_EXECUTED_AND_ANALYZED",
        "scope": "exp282-confirmatory-open-two-phase-result",
        "confirmatory_data_consumed": True,
        "challenge_materialized": False,
        "confirmatory_n": ceremony_seal["confirmatory_n"],
        "reserved_replicate_ids": list(ceremony_seal["reserved_replicate_ids"]),
        "artifacts": {
            "seal": deepcopy(ceremony_seal),
            "raw": deepcopy(raw),
            "analysis": deepcopy(analysis),
        },
        "lineage": {
            "protocol_digest": protocol_digest,
            "ceremony_seal_digest": ceremony_seal["ceremony_seal_digest"],
            "paired_execution_artifact_digest": paired_execution_artifact["artifact_digest"],
            "raw_artifact_digest": raw["artifact_digest"],
            "analysis_digest": analysis["analysis_digest"],
            "paired_checkpoint_sha256": actual_checkpoint_sha,
            "ceremony_code_digest": ceremony_code_digest,
        },
        "remaining_blockers": list(analysis.get("remaining_blockers") or []),
        "ceremony_result_digest": "",
    }
    payload["ceremony_result_digest"] = _result_digest(payload)
    errors = validate_exp282_confirmatory_ceremony_result(payload)
    if errors:
        raise RuntimeError("invalid EXP-282 ceremony result: " + "; ".join(errors))
    return payload
