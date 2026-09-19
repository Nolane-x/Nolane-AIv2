from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

SCHEMA = "EXP330-P02-PARAMETER-GROUP-EXECUTION-IDENTITY-V1"
APPROVED_PREREGISTRATION_DIGEST = "9ca527b7647b46e162b7f7e965f4e4ac74dcf4b1a4b71d4af4ab792331db02bb"
PARENT_RUN_ID = 35417293679
PARENT_ARTIFACT_ID = 10576262539
PARENT_ARTIFACT_ZIP_DIGEST = "d95a36e7305617b3b07232f18072ea99ef766a68161c6dbd9a707545253ae6a9"
PARENT_FINAL_JSON_SHA256 = "f59682e2a08d3fd55437bfa3bfa6f4c3a88e950c4bc8a2ee7526774161be6e7f"
PARENT_FINAL_EVIDENCE_DIGEST = "719b0c35b8538ef379a912448d7539bbdf1ef6e856e036a986b7c64b7a648cd7"
RECONSTRUCTION_RUN_ID = 35345351869
RECONSTRUCTION_ARTIFACT_ID = 10547681681


@dataclass(frozen=True, slots=True)
class Exp330ExecutionIdentity:
    schema: str
    source_commit_sha: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_run_id: int
    parent_final_artifact_id: int
    parent_final_artifact_zip_digest: str
    parent_final_evidence_digest: str
    reconstruction_artifact_id: int
    exp330_execution_digest: str


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def canonical_execution_digest(identity: Exp330ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp330_execution_digest", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def validate_execution_identity(identity: Exp330ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-330 identity schema")
    if len(identity.source_commit_sha) != 40 or len(identity.workflow_sha256) != 64:
        raise ValueError("EXP-330 source/workflow identity")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-330 preregistration identity")
    if identity.parent_run_id != PARENT_RUN_ID or identity.parent_final_artifact_id != PARENT_ARTIFACT_ID:
        raise ValueError("EXP-330 parent run identity")
    if identity.parent_final_artifact_zip_digest != PARENT_ARTIFACT_ZIP_DIGEST:
        raise ValueError("EXP-330 parent artifact identity")
    if identity.parent_final_evidence_digest != PARENT_FINAL_EVIDENCE_DIGEST:
        raise ValueError("EXP-330 parent evidence identity")
    if identity.reconstruction_artifact_id != RECONSTRUCTION_ARTIFACT_ID:
        raise ValueError("EXP-330 reconstruction identity")
    if canonical_execution_digest(identity) != identity.exp330_execution_digest:
        raise ValueError("EXP-330 execution digest")
