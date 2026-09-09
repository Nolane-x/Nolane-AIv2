from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from nolane_ai.protocol.evidence import canonical_sha256
from .exp277_beacon import validate_exp277_beacon_receipt
from .exp277_confirmatory_analysis import (
    build_exp277_confirmatory_analysis,
    validate_exp277_confirmatory_analysis,
)
from .exp277_confirmatory_authorization import validate_exp277_gate_a_seal
from .exp277_confirmatory_executor import (
    execute_exp277_confirmatory_challenge,
    validate_exp277_confirmatory_raw,
)
from .exp277_reconstruction_court import validate_exp277_reconstruction_authorization


SCHEMA = "NLM-EXP-277-CONFIRMATORY-CEREMONY-V1"
EXPERIMENT_ID = "EXP-277"
TEST_ONLY_STATUS = "TEST_ONLY_CEREMONY_COMPLETED"
SCIENTIFIC_STATUS = "CONFIRMATORY_CEREMONY_COMPLETED"


def _ceremony_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("ceremony_digest", None)
    return canonical_sha256(clean)


def _require_mode(
    beacon_receipt: dict[str, Any],
    *,
    test_only: bool,
    arm_scientific_lane: bool,
) -> None:
    if not isinstance(test_only, bool) or not isinstance(arm_scientific_lane, bool):
        raise ValueError("EXP-277 ceremony mode flags must be boolean")
    if test_only and arm_scientific_lane:
        raise RuntimeError("EXP-277 TEST-ONLY mode cannot arm the scientific lane")

    beacon_test_only = beacon_receipt.get("test_only") is True
    if test_only:
        if not beacon_test_only:
            raise RuntimeError("EXP-277 --test-only requires a TEST-ONLY beacon receipt")
        if beacon_receipt.get("scientific_evidence_eligible") is not False:
            raise RuntimeError("EXP-277 TEST-ONLY beacon cannot be scientific evidence")
        return

    if beacon_test_only:
        raise RuntimeError("EXP-277 TEST-ONLY beacon cannot cross into scientific mode")
    if arm_scientific_lane is not True:
        raise RuntimeError("EXP-277 scientific lane is not armed")
    if beacon_receipt.get("scientific_evidence_eligible") is not True:
        raise RuntimeError("EXP-277 armed scientific lane requires an eligible real beacon receipt")


def _validate_chain(
    *,
    seal: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    beacon_receipt: dict[str, Any],
) -> None:
    seal_errors = validate_exp277_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-277 Gate A seal: " + "; ".join(seal_errors))
    reconstruction_errors = validate_exp277_reconstruction_authorization(
        reconstruction_authorization
    )
    if reconstruction_errors:
        raise ValueError(
            "invalid EXP-277 reconstruction authorization: "
            + "; ".join(reconstruction_errors)
        )
    if reconstruction_authorization.get("seal_digest") != seal.get("seal_digest"):
        raise ValueError("EXP-277 ceremony reconstruction/seal lineage mismatch")
    if reconstruction_authorization.get("seal_snapshot") != seal:
        raise ValueError("EXP-277 ceremony reconstruction seal snapshot mismatch")

    authorization = seal.get("authorization_snapshot") or {}
    beacon_errors = validate_exp277_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=authorization.get("freeze_commit_timestamp_utc"),
        checkpoint_seal_created_at_utc=authorization.get("checkpoint_seal_created_at_utc"),
    )
    if beacon_errors:
        raise ValueError("invalid EXP-277 beacon receipt: " + "; ".join(beacon_errors))


def _enforce_test_only_boundary(raw: dict[str, Any], analysis: dict[str, Any]) -> None:
    if raw.get("test_only") is not True:
        raise RuntimeError("EXP-277 TEST-ONLY ceremony raw classification drift")
    if raw.get("scientific_evidence_eligible") is not False:
        raise RuntimeError("EXP-277 TEST-ONLY raw crossed scientific evidence boundary")
    if raw.get("confirmatory_data_consumed") is not False:
        raise RuntimeError("EXP-277 TEST-ONLY raw consumed confirmatory data")
    if analysis.get("test_only") is not True:
        raise RuntimeError("EXP-277 TEST-ONLY analysis classification drift")
    if analysis.get("evidence_level") != "EV-E2" or analysis.get("decision") != "UNVERIFIED":
        raise RuntimeError("EXP-277 TEST-ONLY analysis attempted scientific promotion")
    if analysis.get("scientific_evidence_eligible") is not False:
        raise RuntimeError("EXP-277 TEST-ONLY analysis became scientific evidence")
    if analysis.get("confirmatory_data_consumed") is not False:
        raise RuntimeError("EXP-277 TEST-ONLY analysis consumed confirmatory data")
    if analysis.get("decision_rule_executed") is not False:
        raise RuntimeError("EXP-277 TEST-ONLY analysis executed the scientific decision rule")


def execute_exp277_gate_b_ceremony(
    *,
    seal: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
    test_only: bool,
    arm_scientific_lane: bool,
    raw_publisher: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _validate_chain(
        seal=seal,
        reconstruction_authorization=reconstruction_authorization,
        beacon_receipt=beacon_receipt,
    )
    _require_mode(
        beacon_receipt,
        test_only=test_only,
        arm_scientific_lane=arm_scientific_lane,
    )

    raw = execute_exp277_confirmatory_challenge(
        reconstruction_authorization=reconstruction_authorization,
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    raw_errors = validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        raise RuntimeError("invalid EXP-277 confirmatory raw artifact: " + "; ".join(raw_errors))

    # The unanalyzed raw artifact is the first durable post-beacon evidence. A
    # caller may persist it here so integrity failures and later analysis errors
    # cannot erase the first-valid-attempt record.
    if raw_publisher is not None:
        raw_publisher(raw)

    if raw.get("status") == "INVALID_RUN" or raw.get("decision") == "INVALID_RUN":
        detail = "; ".join(raw.get("integrity_errors") or ["unspecified integrity failure"])
        raise RuntimeError("EXP-277 ceremony stopped at INVALID_RUN before analysis: " + detail)

    authorization = seal.get("authorization_snapshot") or {}
    analysis_code_digest = authorization.get("analysis_code_digest")
    if not isinstance(analysis_code_digest, str) or len(analysis_code_digest) != 64:
        raise RuntimeError("EXP-277 frozen analysis code identity missing from Gate A seal")
    analysis = build_exp277_confirmatory_analysis(
        raw_artifact=raw,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        analysis_code_digest=analysis_code_digest,
    )
    analysis_errors = validate_exp277_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if analysis_errors:
        raise RuntimeError("invalid EXP-277 confirmatory analysis: " + "; ".join(analysis_errors))

    if test_only:
        _enforce_test_only_boundary(raw, analysis)
    else:
        if analysis.get("evidence_level") != "EV-E3":
            raise RuntimeError("EXP-277 armed scientific ceremony did not produce EV-E3 analysis")
        if analysis.get("decision") not in {
            "PROMOTE_TO_NEXT_STAGE",
            "HOLD_UNSTABLE",
            "KILL_SUBSYSTEM",
        }:
            raise RuntimeError("EXP-277 armed scientific ceremony decision invalid")
        if analysis.get("decision_rule_executed") is not True:
            raise RuntimeError("EXP-277 armed scientific ceremony did not execute frozen decision rule")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_lane_armed": bool(arm_scientific_lane),
        "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
        "seal_digest": seal.get("seal_digest"),
        "reconstruction_digest": reconstruction_authorization.get("reconstruction_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get("scientific_identity_digest"),
        "current_source_tree_digest": current_source_tree_digest,
        "executor_code_digest": executor_code_digest,
        "raw": raw,
        "analysis": analysis,
        "ceremony_digest": "",
    }
    payload["ceremony_digest"] = _ceremony_digest(payload)
    errors = validate_exp277_gate_b_ceremony(
        payload,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if errors:
        raise RuntimeError("invalid EXP-277 Gate B ceremony: " + "; ".join(errors))
    return payload


def validate_exp277_gate_b_ceremony(
    payload: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-277 Gate B ceremony must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-277 Gate B ceremony identity")
    if payload.get("ceremony_digest") != _ceremony_digest(payload):
        errors.append("EXP-277 Gate B ceremony digest mismatch")

    raw = payload.get("raw")
    analysis = payload.get("analysis")
    if not isinstance(raw, dict):
        errors.append("EXP-277 Gate B ceremony raw artifact missing")
        return errors
    if not isinstance(analysis, dict):
        errors.append("EXP-277 Gate B ceremony analysis artifact missing")
        return errors
    raw_errors = validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if raw_errors:
        errors.append("embedded EXP-277 raw artifact invalid: " + "; ".join(raw_errors))
        return errors
    analysis_errors = validate_exp277_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if analysis_errors:
        errors.append("embedded EXP-277 analysis invalid: " + "; ".join(analysis_errors))
        return errors

    test_only = payload.get("test_only")
    if test_only is not raw.get("test_only") or test_only is not analysis.get("test_only"):
        errors.append("EXP-277 ceremony TEST-ONLY classification mismatch")
    if test_only is True:
        if payload.get("status") != TEST_ONLY_STATUS:
            errors.append("EXP-277 TEST-ONLY ceremony status drift")
        if payload.get("scientific_lane_armed") is not False:
            errors.append("EXP-277 TEST-ONLY ceremony cannot arm scientific lane")
        if analysis.get("evidence_level") != "EV-E2" or analysis.get("decision") != "UNVERIFIED":
            errors.append("EXP-277 TEST-ONLY ceremony scientific boundary drift")
        if analysis.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-277 TEST-ONLY ceremony cannot create scientific evidence")
        if analysis.get("decision_rule_executed") is not False:
            errors.append("EXP-277 TEST-ONLY ceremony cannot execute scientific decision rule")
    elif test_only is False:
        if payload.get("status") != SCIENTIFIC_STATUS:
            errors.append("EXP-277 scientific ceremony status drift")
        if payload.get("scientific_lane_armed") is not True:
            errors.append("EXP-277 scientific ceremony must be explicitly armed")
        if analysis.get("evidence_level") != "EV-E3":
            errors.append("EXP-277 scientific ceremony analysis evidence level drift")
        if analysis.get("decision") not in {
            "PROMOTE_TO_NEXT_STAGE",
            "HOLD_UNSTABLE",
            "KILL_SUBSYSTEM",
        }:
            errors.append("EXP-277 scientific ceremony decision invalid")
        if analysis.get("decision_rule_executed") is not True:
            errors.append("EXP-277 scientific ceremony must execute frozen decision rule")
    else:
        errors.append("EXP-277 ceremony test_only flag missing")

    raw_lineage = raw.get("lineage") or {}
    if payload.get("beacon_receipt_digest") != raw_lineage.get("beacon_receipt_digest"):
        errors.append("EXP-277 ceremony beacon lineage mismatch")
    if payload.get("seal_digest") != (raw.get("seal") or {}).get("seal_digest"):
        errors.append("EXP-277 ceremony seal lineage mismatch")
    if payload.get("reconstruction_digest") != (
        raw.get("reconstruction_authorization") or {}
    ).get("reconstruction_digest"):
        errors.append("EXP-277 ceremony reconstruction lineage mismatch")
    if payload.get("checkpoint_scientific_identity_digest") != checkpoint_receipt.get(
        "scientific_identity_digest"
    ):
        errors.append("EXP-277 ceremony checkpoint lineage mismatch")
    if payload.get("current_source_tree_digest") != raw_lineage.get("source_tree_digest"):
        errors.append("EXP-277 ceremony source-tree lineage mismatch")
    if payload.get("executor_code_digest") != raw_lineage.get("executor_code_digest"):
        errors.append("EXP-277 ceremony executor lineage mismatch")
    if (analysis.get("lineage") or {}).get("raw_artifact_digest") != raw.get("artifact_digest"):
        errors.append("EXP-277 ceremony raw/analysis digest lineage mismatch")
    return errors