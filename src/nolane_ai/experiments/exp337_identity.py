from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .exp337_contract import (
    APPROVED_PREREGISTRATION_DIGEST,
    PARENT_EXP336_CHUNK7_ARTIFACT_ID,
    PARENT_EXP336_CHUNK7_BUNDLE_DIGEST,
    PARENT_EXP336_CHUNK7_ZIP_SHA256,
    PARENT_EXP336_CLOSURE_DIGEST,
    PARENT_EXP336_EVIDENCE_DIGEST,
    PARENT_EXP336_HEAD_SHA,
    PARENT_EXP336_RUN_ID,
    PARENT_PROJECT_CHECKPOINT_SHA256,
    PARENT_PROJECT_MODEL_STATE_DIGEST,
    PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
    PARENT_PROJECT_RECEIPT_DIGEST,
    PARENT_PROJECT_RECEIPT_JSON_SHA256,
    PARENT_PROJECT_RNG_STATE_DIGEST,
)

SCHEMA = "EXP337-CNRS-SELF-ROLLIN-EXECUTION-IDENTITY-V1"


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
class Exp337ExecutionIdentity:
    schema: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    approved_preregistration_digest: str
    parent_exp336_run_id: int
    parent_exp336_head_sha: str
    parent_exp336_closure_digest: str
    parent_exp336_evidence_digest: str
    parent_chunk7_artifact_id: int
    parent_chunk7_zip_sha256: str
    parent_chunk7_bundle_digest: str
    parent_project_checkpoint_sha256: str
    parent_project_receipt_json_sha256: str
    parent_project_receipt_digest: str
    parent_project_model_state_digest: str
    parent_project_optimizer_state_digest: str
    parent_project_rng_state_digest: str
    exp337_execution_digest: str


def material(identity: Exp337ExecutionIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload.pop("exp337_execution_digest", None)
    return payload


def execution_digest(identity: Exp337ExecutionIdentity) -> str:
    return hashlib.sha256(_canonical(material(identity))).hexdigest()


def validate_execution_identity(identity: Exp337ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-337 identity schema")
    if not _hex(identity.source_commit_sha, 40):
        raise ValueError("EXP-337 source commit")
    if not _hex(identity.source_tree_digest, 64):
        raise ValueError("EXP-337 source tree")
    if not _hex(identity.workflow_sha256, 64):
        raise ValueError("EXP-337 workflow sha")
    if identity.approved_preregistration_digest != APPROVED_PREREGISTRATION_DIGEST:
        raise ValueError("EXP-337 preregistration digest")

    fixed = {
        "parent_exp336_run_id": PARENT_EXP336_RUN_ID,
        "parent_exp336_head_sha": PARENT_EXP336_HEAD_SHA,
        "parent_exp336_closure_digest": PARENT_EXP336_CLOSURE_DIGEST,
        "parent_exp336_evidence_digest": PARENT_EXP336_EVIDENCE_DIGEST,
        "parent_chunk7_artifact_id": PARENT_EXP336_CHUNK7_ARTIFACT_ID,
        "parent_chunk7_zip_sha256": PARENT_EXP336_CHUNK7_ZIP_SHA256,
        "parent_chunk7_bundle_digest": PARENT_EXP336_CHUNK7_BUNDLE_DIGEST,
        "parent_project_checkpoint_sha256": PARENT_PROJECT_CHECKPOINT_SHA256,
        "parent_project_receipt_json_sha256": PARENT_PROJECT_RECEIPT_JSON_SHA256,
        "parent_project_receipt_digest": PARENT_PROJECT_RECEIPT_DIGEST,
        "parent_project_model_state_digest": PARENT_PROJECT_MODEL_STATE_DIGEST,
        "parent_project_optimizer_state_digest": PARENT_PROJECT_OPTIMIZER_STATE_DIGEST,
        "parent_project_rng_state_digest": PARENT_PROJECT_RNG_STATE_DIGEST,
    }
    for key, expected in fixed.items():
        if getattr(identity, key) != expected:
            raise ValueError(f"EXP-337 authority drift: {key}")

    if identity.exp337_execution_digest != execution_digest(identity):
        raise ValueError("EXP-337 execution digest")
