from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .exp333_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_EXP332_ARTIFACT_ID,
    PARENT_EXP332_EVIDENCE_DIGEST,
    PARENT_EXP332_RUN_ID,
    PARENT_EXP332_ZIP_DIGEST,
    RECONSTRUCTION_ARTIFACT_ID,
)

SCHEMA = "EXP333-COMPLETE-8WORLD-PAIR-LATTICE-EXECUTION-IDENTITY-V1"

@dataclass(frozen=True, slots=True)
class Exp333ExecutionIdentity:
    schema: str
    source_commit_sha: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp332_run_id: int
    parent_exp332_artifact_id: int
    parent_exp332_zip_digest: str
    parent_exp332_evidence_digest: str
    reconstruction_artifact_id: int
    exp333_execution_digest: str

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def canonical_execution_digest(identity: Exp333ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp333_execution_digest", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()

def validate_execution_identity(identity: Exp333ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-333 identity schema")
    if len(identity.source_commit_sha) != 40 or len(identity.workflow_sha256) != 64:
        raise ValueError("EXP-333 source/workflow identity")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-333 preregistration identity")
    expected = (
        identity.parent_exp332_run_id == PARENT_EXP332_RUN_ID
        and identity.parent_exp332_artifact_id == PARENT_EXP332_ARTIFACT_ID
        and identity.parent_exp332_zip_digest == PARENT_EXP332_ZIP_DIGEST
        and identity.parent_exp332_evidence_digest == PARENT_EXP332_EVIDENCE_DIGEST
        and identity.reconstruction_artifact_id == RECONSTRUCTION_ARTIFACT_ID
    )
    if not expected:
        raise ValueError("EXP-333 authority identity")
    if canonical_execution_digest(identity) != identity.exp333_execution_digest:
        raise ValueError("EXP-333 execution digest")
