from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp321_contract import (
    EFFORT_GRID,
    EXP319_FINAL_EVIDENCE_DIGEST,
    SELECTED_CHECKPOINT_ARTIFACT_ID,
    SELECTED_CHECKPOINT_ZIP_DIGEST,
    SELECTED_MODEL_STATE_DIGEST,
    SELECTED_RECEIPT_ARTIFACT_DIGEST,
    preregistration_digest,
)


EXP321_IDENTITY_SCHEMA = "EXP321-AFIXED-FAILURE-LOCALIZATION-EXECUTION-IDENTITY-V1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp321ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_exp319_final_evidence_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    selected_checkpoint_artifact_id: int
    selected_checkpoint_zip_digest: str
    selected_receipt_artifact_digest: str
    selected_model_state_digest: str
    effort_grid: tuple[int, ...]
    device: str
    gradient_updates: int
    exp321_execution_digest: str


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
        f"EXP321-GIT-TREE-V1|{git_tree_sha}".encode("ascii")
    ).hexdigest()


def canonical_exp321_execution_digest(identity: Exp321ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp321_execution_digest", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_exp321_execution_identity(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    workflow_sha256: str,
) -> Exp321ExecutionIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(workflow_sha256, length=64, field="workflow sha256")

    values = {
        "schema": EXP321_IDENTITY_SCHEMA,
        "approved_preregistration_digest": preregistration_digest(),
        "parent_exp319_final_evidence_digest": EXP319_FINAL_EVIDENCE_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256": workflow_sha256,
        "selected_checkpoint_artifact_id": SELECTED_CHECKPOINT_ARTIFACT_ID,
        "selected_checkpoint_zip_digest": SELECTED_CHECKPOINT_ZIP_DIGEST,
        "selected_receipt_artifact_digest": SELECTED_RECEIPT_ARTIFACT_DIGEST,
        "selected_model_state_digest": SELECTED_MODEL_STATE_DIGEST,
        "effort_grid": tuple(EFFORT_GRID),
        "device": "cpu",
        "gradient_updates": 0,
    }
    provisional = Exp321ExecutionIdentity(**values, exp321_execution_digest="")
    return Exp321ExecutionIdentity(
        **values,
        exp321_execution_digest=canonical_exp321_execution_digest(provisional),
    )


def canonical_exp321_identity_json_bytes(identity: Exp321ExecutionIdentity) -> bytes:
    if identity.exp321_execution_digest != canonical_exp321_execution_digest(identity):
        raise ValueError("EXP-321 execution digest mismatch")
    return _canonical_bytes(asdict(identity)) + b"\n"
