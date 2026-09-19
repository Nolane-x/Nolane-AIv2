from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from typing import Mapping, Any
from .exp334_contract import (
    APPROVED_PREREGISTRATION_DIGEST, PARENT_EXP333_RUN_ID, PARENT_EXP333_ARTIFACT_ID,
    PARENT_EXP333_ZIP_DIGEST, PARENT_EXP333_EVIDENCE_DIGEST, RECONSTRUCTION_ARTIFACT_ID
)

SCHEMA = "EXP334-HIGHER-ORDER-CONFLICT-SUBSPACE-EXECUTION-IDENTITY-V1"

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

@dataclass(frozen=True, slots=True)
class Exp334ExecutionIdentity:
    schema: str
    source_commit_sha: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp333_run_id: int
    parent_exp333_artifact_id: int
    parent_exp333_zip_digest: str
    parent_exp333_evidence_digest: str
    reconstruction_artifact_id: int
    exp334_execution_digest: str

def material(identity: Exp334ExecutionIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload.pop("exp334_execution_digest", None)
    return payload

def execution_digest(identity: Exp334ExecutionIdentity) -> str:
    return hashlib.sha256(_canonical(material(identity))).hexdigest()

def validate_execution_identity(identity: Exp334ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-334 identity schema")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-334 preregistration")
    if identity.parent_exp333_run_id != PARENT_EXP333_RUN_ID or identity.parent_exp333_artifact_id != PARENT_EXP333_ARTIFACT_ID:
        raise ValueError("EXP-334 parent")
    if identity.parent_exp333_zip_digest != PARENT_EXP333_ZIP_DIGEST or identity.parent_exp333_evidence_digest != PARENT_EXP333_EVIDENCE_DIGEST:
        raise ValueError("EXP-334 parent digest")
    if identity.reconstruction_artifact_id != RECONSTRUCTION_ARTIFACT_ID:
        raise ValueError("EXP-334 reconstruction")
    if not isinstance(identity.source_commit_sha, str) or len(identity.source_commit_sha) != 40:
        raise ValueError("EXP-334 source sha")
    if not isinstance(identity.workflow_sha256, str) or len(identity.workflow_sha256) != 64:
        raise ValueError("EXP-334 workflow sha")
    if identity.exp334_execution_digest != execution_digest(identity):
        raise ValueError("EXP-334 execution digest")
