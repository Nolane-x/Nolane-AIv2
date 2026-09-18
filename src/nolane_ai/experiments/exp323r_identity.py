from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from .exp323_contract import ARMS, CHECKPOINTS, MAX_ADDITIONAL_STEPS, preregistration_digest


SCHEMA = "EXP323R-REPAIRED-EXECUTION-IDENTITY-V1"
PARENT_EXP323_MARKER_SHA = "45b539b5c69c982601ced9d642935472cc8458e1"
FAILED_EXP323_RUN_ID = 35343381108
SELECTION_LOCK_DIGEST = "f74a22e4b3202f5157f0ca71faceedf69cbbf172f04a88aeaf8206f03066d011"
MATERIALIZATION_RUN_ID = 35345351869
SELECTED_ARTIFACT_ID = 10547681681
SELECTED_ARTIFACT_ZIP_DIGEST = "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
SELECTED_CHECKPOINT_SHA256 = "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
SELECTED_RECEIPT_DIGEST = "f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp323RExecutionIdentity:
    schema: str
    approved_exp323_preregistration_digest: str
    parent_exp323_marker_sha: str
    failed_exp323_run_id: int
    failed_exp323_disposition: str
    selection_lock_digest: str
    materialization_run_id: int
    selected_artifact_id: int
    selected_artifact_zip_digest: str
    selected_checkpoint_sha256: str
    selected_receipt_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    arms: tuple[str, ...]
    checkpoints: tuple[int, ...]
    device: str
    max_additional_optimizer_steps_per_arm: int
    exp323r_execution_digest: str


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hex(value: str, length: int, field: str) -> None:
    if len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be {length} lowercase hexadecimal characters")


def source_tree_digest_from_git_tree_sha(tree_sha: str) -> str:
    _hex(tree_sha, 40, "git tree")
    return hashlib.sha256(f"EXP323R-GIT-TREE-V1|{tree_sha}".encode("ascii")).hexdigest()


def execution_digest(identity: Exp323RExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp323r_execution_digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_execution_identity(identity: Exp323RExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-323R identity schema mismatch")
    for value, length, field in (
        (identity.parent_exp323_marker_sha, 40, "parent marker"),
        (identity.selection_lock_digest, 64, "selection lock digest"),
        (identity.selected_artifact_zip_digest, 64, "artifact ZIP digest"),
        (identity.selected_checkpoint_sha256, 64, "checkpoint SHA"),
        (identity.selected_receipt_digest, 64, "receipt digest"),
        (identity.source_commit_sha, 40, "source commit"),
        (identity.source_tree_digest, 64, "source tree digest"),
        (identity.workflow_sha256, 64, "workflow sha256"),
        (identity.exp323r_execution_digest, 64, "execution digest"),
    ):
        _hex(value, length, field)
    if identity.approved_exp323_preregistration_digest != preregistration_digest():
        raise ValueError("EXP-323R preregistration digest mismatch")
    if identity.parent_exp323_marker_sha != PARENT_EXP323_MARKER_SHA:
        raise ValueError("EXP-323R parent marker mismatch")
    if identity.failed_exp323_run_id != FAILED_EXP323_RUN_ID:
        raise ValueError("EXP-323R failed run mismatch")
    if identity.failed_exp323_disposition != "PROCEDURAL_REPLAY_REPRODUCIBILITY_FAILURE_NO_SCIENTIFIC_DISPOSITION":
        raise ValueError("EXP-323R failed-run disposition mismatch")
    if identity.selection_lock_digest != SELECTION_LOCK_DIGEST:
        raise ValueError("EXP-323R selection lock mismatch")
    if identity.materialization_run_id != MATERIALIZATION_RUN_ID:
        raise ValueError("EXP-323R materialization run mismatch")
    if identity.selected_artifact_id != SELECTED_ARTIFACT_ID:
        raise ValueError("EXP-323R selected artifact mismatch")
    if identity.selected_artifact_zip_digest != SELECTED_ARTIFACT_ZIP_DIGEST:
        raise ValueError("EXP-323R selected artifact ZIP mismatch")
    if identity.selected_checkpoint_sha256 != SELECTED_CHECKPOINT_SHA256:
        raise ValueError("EXP-323R checkpoint SHA mismatch")
    if identity.selected_receipt_digest != SELECTED_RECEIPT_DIGEST:
        raise ValueError("EXP-323R receipt digest mismatch")
    if tuple(identity.arms) != tuple(ARMS) or tuple(identity.checkpoints) != tuple(CHECKPOINTS):
        raise ValueError("EXP-323R scientific geometry mismatch")
    if identity.device != "cpu" or identity.max_additional_optimizer_steps_per_arm != MAX_ADDITIONAL_STEPS:
        raise ValueError("EXP-323R runtime geometry mismatch")
    if identity.exp323r_execution_digest != execution_digest(identity):
        raise ValueError("EXP-323R execution digest mismatch")


def build_execution_identity(*, source_commit_sha: str, git_tree_sha: str, workflow_sha256: str) -> Exp323RExecutionIdentity:
    values = {
        "schema": SCHEMA,
        "approved_exp323_preregistration_digest": preregistration_digest(),
        "parent_exp323_marker_sha": PARENT_EXP323_MARKER_SHA,
        "failed_exp323_run_id": FAILED_EXP323_RUN_ID,
        "failed_exp323_disposition": "PROCEDURAL_REPLAY_REPRODUCIBILITY_FAILURE_NO_SCIENTIFIC_DISPOSITION",
        "selection_lock_digest": SELECTION_LOCK_DIGEST,
        "materialization_run_id": MATERIALIZATION_RUN_ID,
        "selected_artifact_id": SELECTED_ARTIFACT_ID,
        "selected_artifact_zip_digest": SELECTED_ARTIFACT_ZIP_DIGEST,
        "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "selected_receipt_digest": SELECTED_RECEIPT_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256": workflow_sha256,
        "arms": tuple(ARMS),
        "checkpoints": tuple(CHECKPOINTS),
        "device": "cpu",
        "max_additional_optimizer_steps_per_arm": MAX_ADDITIONAL_STEPS,
    }
    provisional = Exp323RExecutionIdentity(**values, exp323r_execution_digest="")
    return Exp323RExecutionIdentity(**values, exp323r_execution_digest=execution_digest(provisional))


def canonical_identity_json_bytes(identity: Exp323RExecutionIdentity) -> bytes:
    validate_execution_identity(identity)
    return canonical_json_bytes(asdict(identity)) + b"\n"
