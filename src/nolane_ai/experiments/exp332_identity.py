from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json
from typing import Any, Mapping

from .exp332_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    EXP327_ATTEMPT1_ARTIFACT_ID,
    EXP327_ATTEMPT1_EVIDENCE_DIGEST,
    EXP327_ATTEMPT1_ZIP_DIGEST,
    EXP327_ATTEMPT2_ARTIFACT_ID,
    EXP327_ATTEMPT2_EVIDENCE_DIGEST,
    EXP327_ATTEMPT2_ZIP_DIGEST,
    EXP330_ARTIFACT_ID,
    EXP330_EVIDENCE_DIGEST,
    EXP330_RUN_ID,
    EXP330_ZIP_DIGEST,
    RECONSTRUCTION_ARTIFACT_ID,
)

SCHEMA = "EXP332-PORTABLE-PAIR-LATTICE-EXECUTION-IDENTITY-V1"

@dataclass(frozen=True, slots=True)
class Exp332ExecutionIdentity:
    schema: str
    source_commit_sha: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp330_run_id: int
    parent_exp330_artifact_id: int
    parent_exp330_zip_digest: str
    parent_exp330_evidence_digest: str
    exp327_attempt1_artifact_id: int
    exp327_attempt1_zip_digest: str
    exp327_attempt1_evidence_digest: str
    exp327_attempt2_artifact_id: int
    exp327_attempt2_zip_digest: str
    exp327_attempt2_evidence_digest: str
    reconstruction_artifact_id: int
    exp332_execution_digest: str

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def canonical_execution_digest(identity: Exp332ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp332_execution_digest", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()

def validate_execution_identity(identity: Exp332ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-332 identity schema")
    if len(identity.source_commit_sha) != 40 or len(identity.workflow_sha256) != 64:
        raise ValueError("EXP-332 source/workflow identity")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-332 preregistration identity")
    expected = (
        identity.parent_exp330_run_id == EXP330_RUN_ID
        and identity.parent_exp330_artifact_id == EXP330_ARTIFACT_ID
        and identity.parent_exp330_zip_digest == EXP330_ZIP_DIGEST
        and identity.parent_exp330_evidence_digest == EXP330_EVIDENCE_DIGEST
        and identity.exp327_attempt1_artifact_id == EXP327_ATTEMPT1_ARTIFACT_ID
        and identity.exp327_attempt1_zip_digest == EXP327_ATTEMPT1_ZIP_DIGEST
        and identity.exp327_attempt1_evidence_digest == EXP327_ATTEMPT1_EVIDENCE_DIGEST
        and identity.exp327_attempt2_artifact_id == EXP327_ATTEMPT2_ARTIFACT_ID
        and identity.exp327_attempt2_zip_digest == EXP327_ATTEMPT2_ZIP_DIGEST
        and identity.exp327_attempt2_evidence_digest == EXP327_ATTEMPT2_EVIDENCE_DIGEST
        and identity.reconstruction_artifact_id == RECONSTRUCTION_ARTIFACT_ID
    )
    if not expected:
        raise ValueError("EXP-332 authority identity")
    if canonical_execution_digest(identity) != identity.exp332_execution_digest:
        raise ValueError("EXP-332 execution digest")
