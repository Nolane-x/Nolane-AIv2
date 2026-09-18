from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp325_contract import (
    FAMILIES,
    LOCAL_CHECKPOINTS,
    UPDATES_PER_WORLD,
    WORLD_INDICES,
    preregistration_digest,
)


SCHEMA = "EXP325-SINGLE-WORLD-EXECUTION-IDENTITY-V1"
PARENT_MARKER_SHA = "3bc060f1d3656dbae1de2265a32b145d9cff0dbc"
PARENT_RUN_ID = 35362450626
PARENT_FINAL_ARTIFACT_ID = 10554724648
PARENT_FINAL_ARTIFACT_ZIP_DIGEST = "70fae1e9d8b54df17f2a46740630cbb1861662a5ec3d2df756193e3cd6991965"
PARENT_FINAL_EVIDENCE_DIGEST = "10a493d2f4de15131706b933af4170f39f55e802f069e740f9f98846713ac2ad"
PARENT_EXECUTION_DIGEST = "fbf1e416f3d195a44ca901ed8e2c7377c6850e4e32ec0116bb59d7657513f2f0"
RECONSTRUCTION_RUN_ID = 35345351869
RECONSTRUCTION_ARTIFACT_ID = 10547681681
RECONSTRUCTION_ARTIFACT_ZIP_DIGEST = "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
RECONSTRUCTION_CHECKPOINT_SHA256 = "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
RECONSTRUCTION_RECEIPT_DIGEST = "f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp325ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_marker_sha: str
    parent_run_id: int
    parent_final_artifact_id: int
    parent_final_artifact_zip_digest: str
    parent_final_evidence_digest: str
    parent_execution_digest: str
    reconstruction_run_id: int
    reconstruction_artifact_id: int
    reconstruction_artifact_zip_digest: str
    reconstruction_checkpoint_sha256: str
    reconstruction_receipt_digest: str
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    families: tuple[str, ...]
    world_indices: tuple[int, ...]
    local_checkpoints: tuple[int, ...]
    updates_per_world: int
    device: str
    exp325_execution_digest: str


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hex(value: str, length: int, field: str) -> None:
    if len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be {length} lowercase hex characters")


def source_tree_digest_from_git_tree_sha(tree_sha: str) -> str:
    _hex(tree_sha, 40, "git tree")
    return hashlib.sha256(f"EXP325-GIT-TREE-V1|{tree_sha}".encode("ascii")).hexdigest()


def execution_digest(identity: Exp325ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp325_execution_digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_execution_identity(identity: Exp325ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-325 identity schema mismatch")
    for value, length, field in (
        (identity.parent_marker_sha, 40, "parent marker"),
        (identity.parent_final_artifact_zip_digest, 64, "parent ZIP"),
        (identity.parent_final_evidence_digest, 64, "parent evidence"),
        (identity.parent_execution_digest, 64, "parent execution"),
        (identity.reconstruction_artifact_zip_digest, 64, "reconstruction ZIP"),
        (identity.reconstruction_checkpoint_sha256, 64, "checkpoint SHA"),
        (identity.reconstruction_receipt_digest, 64, "receipt digest"),
        (identity.source_commit_sha, 40, "source commit"),
        (identity.source_tree_digest, 64, "source tree"),
        (identity.workflow_sha256, 64, "workflow SHA"),
        (identity.exp325_execution_digest, 64, "execution digest"),
    ):
        _hex(value, length, field)
    if identity.approved_preregistration_digest != preregistration_digest():
        raise ValueError("EXP-325 preregistration mismatch")
    if identity.parent_marker_sha != PARENT_MARKER_SHA:
        raise ValueError("EXP-325 parent marker mismatch")
    if identity.parent_run_id != PARENT_RUN_ID:
        raise ValueError("EXP-325 parent run mismatch")
    if (
        identity.parent_final_artifact_id != PARENT_FINAL_ARTIFACT_ID
        or identity.parent_final_artifact_zip_digest != PARENT_FINAL_ARTIFACT_ZIP_DIGEST
        or identity.parent_final_evidence_digest != PARENT_FINAL_EVIDENCE_DIGEST
        or identity.parent_execution_digest != PARENT_EXECUTION_DIGEST
    ):
        raise ValueError("EXP-325 parent authority mismatch")
    if (
        identity.reconstruction_run_id != RECONSTRUCTION_RUN_ID
        or identity.reconstruction_artifact_id != RECONSTRUCTION_ARTIFACT_ID
        or identity.reconstruction_artifact_zip_digest != RECONSTRUCTION_ARTIFACT_ZIP_DIGEST
        or identity.reconstruction_checkpoint_sha256 != RECONSTRUCTION_CHECKPOINT_SHA256
        or identity.reconstruction_receipt_digest != RECONSTRUCTION_RECEIPT_DIGEST
    ):
        raise ValueError("EXP-325 reconstruction authority mismatch")
    if tuple(identity.families) != tuple(FAMILIES):
        raise ValueError("EXP-325 family geometry mismatch")
    if tuple(identity.world_indices) != tuple(WORLD_INDICES):
        raise ValueError("EXP-325 world-index geometry mismatch")
    if tuple(identity.local_checkpoints) != tuple(LOCAL_CHECKPOINTS):
        raise ValueError("EXP-325 checkpoint geometry mismatch")
    if identity.updates_per_world != UPDATES_PER_WORLD or identity.device != "cpu":
        raise ValueError("EXP-325 runtime geometry mismatch")
    if identity.exp325_execution_digest != execution_digest(identity):
        raise ValueError("EXP-325 execution digest mismatch")


def build_execution_identity(*, source_commit_sha: str, git_tree_sha: str, workflow_sha256: str) -> Exp325ExecutionIdentity:
    values = {
        "schema": SCHEMA,
        "approved_preregistration_digest": preregistration_digest(),
        "parent_marker_sha": PARENT_MARKER_SHA,
        "parent_run_id": PARENT_RUN_ID,
        "parent_final_artifact_id": PARENT_FINAL_ARTIFACT_ID,
        "parent_final_artifact_zip_digest": PARENT_FINAL_ARTIFACT_ZIP_DIGEST,
        "parent_final_evidence_digest": PARENT_FINAL_EVIDENCE_DIGEST,
        "parent_execution_digest": PARENT_EXECUTION_DIGEST,
        "reconstruction_run_id": RECONSTRUCTION_RUN_ID,
        "reconstruction_artifact_id": RECONSTRUCTION_ARTIFACT_ID,
        "reconstruction_artifact_zip_digest": RECONSTRUCTION_ARTIFACT_ZIP_DIGEST,
        "reconstruction_checkpoint_sha256": RECONSTRUCTION_CHECKPOINT_SHA256,
        "reconstruction_receipt_digest": RECONSTRUCTION_RECEIPT_DIGEST,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest_from_git_tree_sha(git_tree_sha),
        "workflow_sha256": workflow_sha256,
        "families": tuple(FAMILIES),
        "world_indices": tuple(WORLD_INDICES),
        "local_checkpoints": tuple(LOCAL_CHECKPOINTS),
        "updates_per_world": UPDATES_PER_WORLD,
        "device": "cpu",
    }
    provisional = Exp325ExecutionIdentity(**values, exp325_execution_digest="")
    return Exp325ExecutionIdentity(
        **values, exp325_execution_digest=execution_digest(provisional)
    )


def canonical_identity_json_bytes(identity: Exp325ExecutionIdentity) -> bytes:
    validate_execution_identity(identity)
    return canonical_json_bytes(asdict(identity)) + b"\n"
