from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp322_contract import (
    ARMS,
    CHECKPOINTS,
    MAX_ADDITIONAL_STEPS,
    PARENT_CHECKPOINT_ARTIFACT_ID,
    PARENT_CHECKPOINT_ZIP_DIGEST,
    PARENT_EXP321_EVIDENCE_DIGEST,
    PARENT_MODEL_STATE_DIGEST,
    PARENT_RECEIPT_ARTIFACT_DIGEST,
    preregistration_digest,
)

EXP322_IDENTITY_SCHEMA = "EXP322-TEACHER-FORCED-INTERVENTION-EXECUTION-IDENTITY-V1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp322ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_exp321_evidence_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    selected_checkpoint_artifact_id: int
    selected_checkpoint_zip_digest: str
    selected_receipt_artifact_digest: str
    selected_model_state_digest: str
    arms: tuple[str, ...]
    checkpoints: tuple[int, ...]
    device: str
    max_additional_optimizer_steps_per_arm: int
    exp322_execution_digest: str


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _require_hex(value: str, *, length: int, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX for ch in value)
    ):
        raise ValueError(
            f"{field} must be exactly {length} lowercase hexadecimal characters"
        )


def source_tree_digest_from_git_tree_sha(git_tree_sha: str) -> str:
    _require_hex(git_tree_sha, length=40, field="git tree")
    return hashlib.sha256(
        f"EXP322-GIT-TREE-V1|{git_tree_sha}".encode("ascii")
    ).hexdigest()


def canonical_exp322_execution_digest(identity: Exp322ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp322_execution_digest", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_exp322_execution_identity(identity: Exp322ExecutionIdentity) -> None:
    if identity.schema != EXP322_IDENTITY_SCHEMA:
        raise ValueError("EXP-322 execution identity schema mismatch")
    _require_hex(identity.source_commit_sha, length=40, field="source commit")
    _require_hex(identity.source_tree_digest, length=64, field="source tree digest")
    _require_hex(identity.workflow_sha256, length=64, field="workflow sha256")
    _require_hex(identity.exp322_execution_digest, length=64, field="execution digest")
    if identity.approved_preregistration_digest != preregistration_digest():
        raise ValueError("EXP-322 preregistration digest mismatch")
    if identity.parent_exp321_evidence_digest != PARENT_EXP321_EVIDENCE_DIGEST:
        raise ValueError("EXP-322 parent EXP-321 evidence mismatch")
    if identity.selected_checkpoint_artifact_id != PARENT_CHECKPOINT_ARTIFACT_ID:
        raise ValueError("EXP-322 selected checkpoint artifact mismatch")
    if identity.selected_checkpoint_zip_digest != PARENT_CHECKPOINT_ZIP_DIGEST:
        raise ValueError("EXP-322 selected checkpoint ZIP mismatch")
    if identity.selected_receipt_artifact_digest != PARENT_RECEIPT_ARTIFACT_DIGEST:
        raise ValueError("EXP-322 selected receipt mismatch")
    if identity.selected_model_state_digest != PARENT_MODEL_STATE_DIGEST:
        raise ValueError("EXP-322 selected model-state mismatch")
    if tuple(identity.arms) != tuple(ARMS):
        raise ValueError("EXP-322 arm geometry mismatch")
    if tuple(identity.checkpoints) != tuple(CHECKPOINTS):
        raise ValueError("EXP-322 checkpoint geometry mismatch")
    if identity.device != "cpu":
        raise ValueError("EXP-322 execution device must remain cpu")
    if identity.max_additional_optimizer_steps_per_arm != MAX_ADDITIONAL_STEPS:
        raise ValueError("EXP-322 optimizer-step ceiling mismatch")
    if identity.exp322_execution_digest != canonical_exp322_execution_digest(identity):
        raise ValueError("EXP-322 execution digest mismatch")


def build_exp322_execution_identity(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    workflow_sha256: str,
) -> Exp322ExecutionIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(workflow_sha256, length=64, field="workflow sha256")
    values = {
        "schema": EXP322_IDENTITY_SCHEMA,
        "approved_preregistration_digest": preregistration_digest(),
        "parent_exp321_evidence_digest": PARENT_EXP321_EVIDENCE_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256": workflow_sha256,
        "selected_checkpoint_artifact_id": PARENT_CHECKPOINT_ARTIFACT_ID,
        "selected_checkpoint_zip_digest": PARENT_CHECKPOINT_ZIP_DIGEST,
        "selected_receipt_artifact_digest": PARENT_RECEIPT_ARTIFACT_DIGEST,
        "selected_model_state_digest": PARENT_MODEL_STATE_DIGEST,
        "arms": tuple(ARMS),
        "checkpoints": tuple(CHECKPOINTS),
        "device": "cpu",
        "max_additional_optimizer_steps_per_arm": MAX_ADDITIONAL_STEPS,
    }
    provisional = Exp322ExecutionIdentity(**values, exp322_execution_digest="")
    return Exp322ExecutionIdentity(
        **values,
        exp322_execution_digest=canonical_exp322_execution_digest(provisional),
    )


def canonical_exp322_identity_json_bytes(identity: Exp322ExecutionIdentity) -> bytes:
    validate_exp322_execution_identity(identity)
    return _canonical_bytes(asdict(identity)) + b"\n"
