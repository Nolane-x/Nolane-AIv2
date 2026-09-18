from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp323_contract import (
    ARMS,
    CHECKPOINTS,
    MAX_ADDITIONAL_STEPS,
    PARENT_DECAY_ARM_EVIDENCE_DIGEST,
    PARENT_DECAY_MODEL_STATE_DIGEST,
    PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
    PARENT_DECAY_RNG_STATE_DIGEST,
    PARENT_EXP322_EVIDENCE_DIGEST,
    PARENT_EXP322_FINAL_ARTIFACT_ID,
    PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST,
    PARENT_EXP322_RUN_ID,
    SOURCE_CHECKPOINT_ARTIFACT_ID,
    SOURCE_CHECKPOINT_ZIP_DIGEST,
    preregistration_digest,
)


EXP323_IDENTITY_SCHEMA = "EXP323-SECOND-DECAY-CONVERGENCE-EXECUTION-IDENTITY-V1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp323ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_exp322_run_id: int
    parent_exp322_final_artifact_id: int
    parent_exp322_final_artifact_zip_digest: str
    parent_exp322_evidence_digest: str
    parent_decay_arm_evidence_digest: str
    parent_decay_model_state_digest: str
    parent_decay_optimizer_state_digest: str
    parent_decay_rng_state_digest: str
    source_checkpoint_artifact_id: int
    source_checkpoint_zip_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    arms: tuple[str, ...]
    checkpoints: tuple[int, ...]
    device: str
    max_additional_optimizer_steps_per_arm: int
    exp323_execution_digest: str


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
        f"EXP323-GIT-TREE-V1|{git_tree_sha}".encode("ascii")
    ).hexdigest()


def canonical_exp323_execution_digest(identity: Exp323ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp323_execution_digest", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_exp323_execution_identity(identity: Exp323ExecutionIdentity) -> None:
    if identity.schema != EXP323_IDENTITY_SCHEMA:
        raise ValueError("EXP-323 execution identity schema mismatch")
    _require_hex(identity.source_commit_sha, length=40, field="source commit")
    _require_hex(identity.source_tree_digest, length=64, field="source tree digest")
    _require_hex(identity.workflow_sha256, length=64, field="workflow sha256")
    _require_hex(identity.exp323_execution_digest, length=64, field="execution digest")
    if identity.approved_preregistration_digest != preregistration_digest():
        raise ValueError("EXP-323 preregistration digest mismatch")
    if identity.parent_exp322_run_id != PARENT_EXP322_RUN_ID:
        raise ValueError("EXP-323 parent run mismatch")
    if identity.parent_exp322_final_artifact_id != PARENT_EXP322_FINAL_ARTIFACT_ID:
        raise ValueError("EXP-323 parent artifact mismatch")
    if (
        identity.parent_exp322_final_artifact_zip_digest
        != PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST
    ):
        raise ValueError("EXP-323 parent artifact ZIP mismatch")
    if identity.parent_exp322_evidence_digest != PARENT_EXP322_EVIDENCE_DIGEST:
        raise ValueError("EXP-323 parent evidence mismatch")
    if identity.parent_decay_arm_evidence_digest != PARENT_DECAY_ARM_EVIDENCE_DIGEST:
        raise ValueError("EXP-323 parent DECAY arm evidence mismatch")
    if identity.parent_decay_model_state_digest != PARENT_DECAY_MODEL_STATE_DIGEST:
        raise ValueError("EXP-323 parent DECAY model-state mismatch")
    if (
        identity.parent_decay_optimizer_state_digest
        != PARENT_DECAY_OPTIMIZER_STATE_DIGEST
    ):
        raise ValueError("EXP-323 parent DECAY optimizer-state mismatch")
    if identity.parent_decay_rng_state_digest != PARENT_DECAY_RNG_STATE_DIGEST:
        raise ValueError("EXP-323 parent DECAY RNG-state mismatch")
    if identity.source_checkpoint_artifact_id != SOURCE_CHECKPOINT_ARTIFACT_ID:
        raise ValueError("EXP-323 source checkpoint artifact mismatch")
    if identity.source_checkpoint_zip_digest != SOURCE_CHECKPOINT_ZIP_DIGEST:
        raise ValueError("EXP-323 source checkpoint ZIP mismatch")
    if tuple(identity.arms) != tuple(ARMS):
        raise ValueError("EXP-323 arm geometry mismatch")
    if tuple(identity.checkpoints) != tuple(CHECKPOINTS):
        raise ValueError("EXP-323 checkpoint geometry mismatch")
    if identity.device != "cpu":
        raise ValueError("EXP-323 execution device must remain cpu")
    if identity.max_additional_optimizer_steps_per_arm != MAX_ADDITIONAL_STEPS:
        raise ValueError("EXP-323 optimizer-step ceiling mismatch")
    if identity.exp323_execution_digest != canonical_exp323_execution_digest(identity):
        raise ValueError("EXP-323 execution digest mismatch")


def build_exp323_execution_identity(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    workflow_sha256: str,
) -> Exp323ExecutionIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(workflow_sha256, length=64, field="workflow sha256")
    values = {
        "schema": EXP323_IDENTITY_SCHEMA,
        "approved_preregistration_digest": preregistration_digest(),
        "parent_exp322_run_id": PARENT_EXP322_RUN_ID,
        "parent_exp322_final_artifact_id": PARENT_EXP322_FINAL_ARTIFACT_ID,
        "parent_exp322_final_artifact_zip_digest": PARENT_EXP322_FINAL_ARTIFACT_ZIP_DIGEST,
        "parent_exp322_evidence_digest": PARENT_EXP322_EVIDENCE_DIGEST,
        "parent_decay_arm_evidence_digest": PARENT_DECAY_ARM_EVIDENCE_DIGEST,
        "parent_decay_model_state_digest": PARENT_DECAY_MODEL_STATE_DIGEST,
        "parent_decay_optimizer_state_digest": PARENT_DECAY_OPTIMIZER_STATE_DIGEST,
        "parent_decay_rng_state_digest": PARENT_DECAY_RNG_STATE_DIGEST,
        "source_checkpoint_artifact_id": SOURCE_CHECKPOINT_ARTIFACT_ID,
        "source_checkpoint_zip_digest": SOURCE_CHECKPOINT_ZIP_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256": workflow_sha256,
        "arms": tuple(ARMS),
        "checkpoints": tuple(CHECKPOINTS),
        "device": "cpu",
        "max_additional_optimizer_steps_per_arm": MAX_ADDITIONAL_STEPS,
    }
    provisional = Exp323ExecutionIdentity(**values, exp323_execution_digest="")
    return Exp323ExecutionIdentity(
        **values,
        exp323_execution_digest=canonical_exp323_execution_digest(provisional),
    )


def canonical_exp323_identity_json_bytes(identity: Exp323ExecutionIdentity) -> bytes:
    validate_exp323_execution_identity(identity)
    return _canonical_bytes(asdict(identity)) + b"\n"
