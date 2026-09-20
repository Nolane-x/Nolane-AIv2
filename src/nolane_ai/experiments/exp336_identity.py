from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .exp336_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_CNRS_ARTIFACT_ID,
    PARENT_CNRS_CHECKPOINT_SHA256,
    PARENT_CNRS_MODEL_STATE_DIGEST,
    PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
    PARENT_CNRS_RECEIPT_SHA256,
    PARENT_CNRS_RNG_STATE_DIGEST,
    PARENT_CNRS_SUMMARY_SHA256,
    PARENT_CNRS_ZIP_DIGEST,
    PARENT_EXP319_RUN_ID,
    PARENT_EXP335_AUDIT_ARTIFACT_ID,
    PARENT_EXP335_AUDIT_REPORT_SHA256,
    PARENT_EXP335_AUDIT_RUN_ID,
    PARENT_EXP335_AUDIT_ZIP_DIGEST,
    PARENT_EXP335_EVIDENCE_DIGEST,
    PARENT_EXP335_EXECUTION_DIGEST,
    PARENT_EXP335_FINAL_ARTIFACT_ID,
    PARENT_EXP335_FINAL_JSON_SHA256,
    PARENT_EXP335_FINAL_ZIP_DIGEST,
    PARENT_EXP335_RUN_ID,
    PARENT_EXP335_SCIENTIFIC_SOURCE_SHA,
    PARENT_EXP335_SEALED_MARKER_SHA,
    PARENT_SELECTION_ARTIFACT_ID,
    PARENT_SELECTION_AUTHORITY_DIGEST,
    PARENT_SELECTION_JSON_SHA256,
    PARENT_SELECTION_ZIP_DIGEST,
)

SCHEMA = "EXP336-CNRS-FULL32-STAGE-A-EXECUTION-IDENTITY-V1"


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hex(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Exp336ExecutionIdentity:
    schema: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp319_run_id: int
    parent_selection_artifact_id: int
    parent_selection_zip_digest: str
    parent_selection_json_sha256: str
    parent_selection_authority_digest: str
    parent_cnrs_artifact_id: int
    parent_cnrs_zip_digest: str
    parent_cnrs_checkpoint_sha256: str
    parent_cnrs_receipt_sha256: str
    parent_cnrs_summary_sha256: str
    parent_cnrs_model_state_digest: str
    parent_cnrs_optimizer_state_digest: str
    parent_cnrs_rng_state_digest: str
    parent_exp335_run_id: int
    parent_exp335_sealed_marker_sha: str
    parent_exp335_scientific_source_sha: str
    parent_exp335_execution_digest: str
    parent_exp335_final_artifact_id: int
    parent_exp335_final_zip_digest: str
    parent_exp335_final_json_sha256: str
    parent_exp335_evidence_digest: str
    parent_exp335_audit_run_id: int
    parent_exp335_audit_artifact_id: int
    parent_exp335_audit_zip_digest: str
    parent_exp335_audit_report_sha256: str
    exp336_execution_digest: str


def material(identity: Exp336ExecutionIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload.pop("exp336_execution_digest", None)
    return payload


def execution_digest(identity: Exp336ExecutionIdentity) -> str:
    return hashlib.sha256(_canonical(material(identity))).hexdigest()


def validate_execution_identity(identity: Exp336ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-336 identity schema")
    if not _hex(identity.source_commit_sha, 40):
        raise ValueError("EXP-336 source commit")
    if not _hex(identity.source_tree_digest, 64):
        raise ValueError("EXP-336 source tree")
    if not _hex(identity.workflow_sha256, 64):
        raise ValueError("EXP-336 workflow sha")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-336 preregistration")

    fixed = {
        "parent_exp319_run_id": PARENT_EXP319_RUN_ID,
        "parent_selection_artifact_id": PARENT_SELECTION_ARTIFACT_ID,
        "parent_selection_zip_digest": PARENT_SELECTION_ZIP_DIGEST,
        "parent_selection_json_sha256": PARENT_SELECTION_JSON_SHA256,
        "parent_selection_authority_digest": PARENT_SELECTION_AUTHORITY_DIGEST,
        "parent_cnrs_artifact_id": PARENT_CNRS_ARTIFACT_ID,
        "parent_cnrs_zip_digest": PARENT_CNRS_ZIP_DIGEST,
        "parent_cnrs_checkpoint_sha256": PARENT_CNRS_CHECKPOINT_SHA256,
        "parent_cnrs_receipt_sha256": PARENT_CNRS_RECEIPT_SHA256,
        "parent_cnrs_summary_sha256": PARENT_CNRS_SUMMARY_SHA256,
        "parent_cnrs_model_state_digest": PARENT_CNRS_MODEL_STATE_DIGEST,
        "parent_cnrs_optimizer_state_digest": PARENT_CNRS_OPTIMIZER_STATE_DIGEST,
        "parent_cnrs_rng_state_digest": PARENT_CNRS_RNG_STATE_DIGEST,
        "parent_exp335_run_id": PARENT_EXP335_RUN_ID,
        "parent_exp335_sealed_marker_sha": PARENT_EXP335_SEALED_MARKER_SHA,
        "parent_exp335_scientific_source_sha": PARENT_EXP335_SCIENTIFIC_SOURCE_SHA,
        "parent_exp335_execution_digest": PARENT_EXP335_EXECUTION_DIGEST,
        "parent_exp335_final_artifact_id": PARENT_EXP335_FINAL_ARTIFACT_ID,
        "parent_exp335_final_zip_digest": PARENT_EXP335_FINAL_ZIP_DIGEST,
        "parent_exp335_final_json_sha256": PARENT_EXP335_FINAL_JSON_SHA256,
        "parent_exp335_evidence_digest": PARENT_EXP335_EVIDENCE_DIGEST,
        "parent_exp335_audit_run_id": PARENT_EXP335_AUDIT_RUN_ID,
        "parent_exp335_audit_artifact_id": PARENT_EXP335_AUDIT_ARTIFACT_ID,
        "parent_exp335_audit_zip_digest": PARENT_EXP335_AUDIT_ZIP_DIGEST,
        "parent_exp335_audit_report_sha256": PARENT_EXP335_AUDIT_REPORT_SHA256,
    }
    for key, expected in fixed.items():
        if getattr(identity, key) != expected:
            raise ValueError(f"EXP-336 authority drift: {key}")
    if identity.exp336_execution_digest != execution_digest(identity):
        raise ValueError("EXP-336 execution digest")
