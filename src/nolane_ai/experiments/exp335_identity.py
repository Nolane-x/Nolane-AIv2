from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .exp335_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_EXP334_ARTIFACT_ID,
    PARENT_EXP334_EVIDENCE_DIGEST,
    PARENT_EXP334_JSON_SHA256,
    PARENT_EXP334_RUN_ID,
    PARENT_EXP334_ZIP_DIGEST,
    RECONSTRUCTION_ARTIFACT_ID,
    RECONSTRUCTION_CHECKPOINT_SHA256,
    RECONSTRUCTION_RECEIPT_SHA256,
    RECONSTRUCTION_RUN_ID,
    RECONSTRUCTION_ZIP_DIGEST,
)

SCHEMA = "EXP335-FULL32-AFIXED-FOUNDATION-REENTRY-EXECUTION-IDENTITY-V1"


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
class Exp335ExecutionIdentity:
    schema: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp334_run_id: int
    parent_exp334_artifact_id: int
    parent_exp334_zip_digest: str
    parent_exp334_final_json_sha256: str
    parent_exp334_evidence_digest: str
    reconstruction_run_id: int
    reconstruction_artifact_id: int
    reconstruction_zip_digest: str
    reconstruction_checkpoint_sha256: str
    reconstruction_receipt_sha256: str
    exp335_execution_digest: str


def material(identity: Exp335ExecutionIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload.pop("exp335_execution_digest", None)
    return payload


def execution_digest(identity: Exp335ExecutionIdentity) -> str:
    return hashlib.sha256(_canonical(material(identity))).hexdigest()


def validate_execution_identity(identity: Exp335ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-335 identity schema")
    if not _hex(identity.source_commit_sha, 40):
        raise ValueError("EXP-335 source commit")
    if not _hex(identity.source_tree_digest, 64):
        raise ValueError("EXP-335 source tree")
    if not _hex(identity.workflow_sha256, 64):
        raise ValueError("EXP-335 workflow sha")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-335 preregistration")
    fixed = {
        "parent_exp334_run_id": PARENT_EXP334_RUN_ID,
        "parent_exp334_artifact_id": PARENT_EXP334_ARTIFACT_ID,
        "parent_exp334_zip_digest": PARENT_EXP334_ZIP_DIGEST,
        "parent_exp334_final_json_sha256": PARENT_EXP334_JSON_SHA256,
        "parent_exp334_evidence_digest": PARENT_EXP334_EVIDENCE_DIGEST,
        "reconstruction_run_id": RECONSTRUCTION_RUN_ID,
        "reconstruction_artifact_id": RECONSTRUCTION_ARTIFACT_ID,
        "reconstruction_zip_digest": RECONSTRUCTION_ZIP_DIGEST,
        "reconstruction_checkpoint_sha256": RECONSTRUCTION_CHECKPOINT_SHA256,
        "reconstruction_receipt_sha256": RECONSTRUCTION_RECEIPT_SHA256,
    }
    for key, expected in fixed.items():
        if getattr(identity, key) != expected:
            raise ValueError(f"EXP-335 authority drift: {key}")
    if identity.exp335_execution_digest != execution_digest(identity):
        raise ValueError("EXP-335 execution digest")
