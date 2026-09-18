from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp326_contract import (
    EXPOSURE_CHECKPOINTS,
    EXPOSURES_PER_WORLD,
    FAMILIES,
    GROUP_IDS,
    preregistration_digest,
)


SCHEMA = "EXP326-BALANCED-MULTIWORLD-EXECUTION-IDENTITY-V1"
PARENT_MARKER_SHA = "b24830c842258258f42ffb92e5f00277f8db8e9b"
PARENT_RUN_ID = 35406123945
PARENT_FINAL_ARTIFACT_ID = 10572536792
PARENT_FINAL_ARTIFACT_ZIP_DIGEST = "e6b1cda21aa84351ec0d61fd4e8a0bfab84177fc5faa5c8ea169030f5badcd32"
PARENT_FINAL_JSON_SHA256 = "bbf4aca8cae39ff424ad2fa4cea0c122a60b405162198413fb921dd891dcaf3c"
PARENT_FINAL_EVIDENCE_DIGEST = "0ce6a8bd4f96864bf754b22fc7c9b95552f6081225a4c305394d92ed39e7cbc2"
PARENT_EXECUTION_DIGEST = "5c3ebd63682b7678cfd40545ef2c2bd6f98de539a7d931f6e47faa8ed21e7f5e"
RECONSTRUCTION_RUN_ID = 35345351869
RECONSTRUCTION_ARTIFACT_ID = 10547681681
RECONSTRUCTION_ARTIFACT_ZIP_DIGEST = "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621"
RECONSTRUCTION_CHECKPOINT_SHA256 = "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
RECONSTRUCTION_RECEIPT_DIGEST = "f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class Exp326ExecutionIdentity:
    schema: str
    approved_preregistration_digest: str
    parent_marker_sha: str
    parent_run_id: int
    parent_final_artifact_id: int
    parent_final_artifact_zip_digest: str
    parent_final_json_sha256: str
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
    group_ids: tuple[str, ...]
    exposure_checkpoints: tuple[int, ...]
    exposures_per_world: int
    device: str
    exp326_execution_digest: str


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hex(value: str, length: int, field: str) -> None:
    if len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be {length} lowercase hex characters")


def source_tree_digest_from_git_tree_sha(tree_sha: str) -> str:
    _hex(tree_sha, 40, "git tree")
    return hashlib.sha256(f"EXP326-GIT-TREE-V1|{tree_sha}".encode("ascii")).hexdigest()


def execution_digest(identity: Exp326ExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("exp326_execution_digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_execution_identity(identity: Exp326ExecutionIdentity) -> None:
    if identity.schema != SCHEMA:
        raise ValueError("EXP-326 identity schema mismatch")
    for value, length, field in (
        (identity.parent_marker_sha, 40, "parent marker"),
        (identity.parent_final_artifact_zip_digest, 64, "parent ZIP"),
        (identity.parent_final_json_sha256, 64, "parent final JSON"),
        (identity.parent_final_evidence_digest, 64, "parent evidence"),
        (identity.parent_execution_digest, 64, "parent execution"),
        (identity.reconstruction_artifact_zip_digest, 64, "reconstruction ZIP"),
        (identity.reconstruction_checkpoint_sha256, 64, "checkpoint SHA"),
        (identity.reconstruction_receipt_digest, 64, "receipt digest"),
        (identity.source_commit_sha, 40, "source commit"),
        (identity.source_tree_digest, 64, "source tree"),
        (identity.workflow_sha256, 64, "workflow SHA"),
        (identity.exp326_execution_digest, 64, "execution digest"),
    ):
        _hex(value, length, field)
    if identity.approved_preregistration_digest != preregistration_digest():
        raise ValueError("EXP-326 preregistration mismatch")
    if identity.parent_marker_sha != PARENT_MARKER_SHA or identity.parent_run_id != PARENT_RUN_ID:
        raise ValueError("EXP-326 parent lineage mismatch")
    if (
        identity.parent_final_artifact_id != PARENT_FINAL_ARTIFACT_ID
        or identity.parent_final_artifact_zip_digest != PARENT_FINAL_ARTIFACT_ZIP_DIGEST
        or identity.parent_final_json_sha256 != PARENT_FINAL_JSON_SHA256
        or identity.parent_final_evidence_digest != PARENT_FINAL_EVIDENCE_DIGEST
        or identity.parent_execution_digest != PARENT_EXECUTION_DIGEST
    ):
        raise ValueError("EXP-326 parent authority mismatch")
    if (
        identity.reconstruction_run_id != RECONSTRUCTION_RUN_ID
        or identity.reconstruction_artifact_id != RECONSTRUCTION_ARTIFACT_ID
        or identity.reconstruction_artifact_zip_digest != RECONSTRUCTION_ARTIFACT_ZIP_DIGEST
        or identity.reconstruction_checkpoint_sha256 != RECONSTRUCTION_CHECKPOINT_SHA256
        or identity.reconstruction_receipt_digest != RECONSTRUCTION_RECEIPT_DIGEST
    ):
        raise ValueError("EXP-326 reconstruction authority mismatch")
    if tuple(identity.families) != tuple(FAMILIES):
        raise ValueError("EXP-326 family geometry mismatch")
    if tuple(identity.group_ids) != tuple(GROUP_IDS):
        raise ValueError("EXP-326 group geometry mismatch")
    if tuple(identity.exposure_checkpoints) != tuple(EXPOSURE_CHECKPOINTS):
        raise ValueError("EXP-326 checkpoint geometry mismatch")
    if identity.exposures_per_world != EXPOSURES_PER_WORLD or identity.device != "cpu":
        raise ValueError("EXP-326 runtime geometry mismatch")
    if identity.exp326_execution_digest != execution_digest(identity):
        raise ValueError("EXP-326 execution digest mismatch")


def build_execution_identity(*, source_commit_sha: str, git_tree_sha: str, workflow_sha256: str) -> Exp326ExecutionIdentity:
    values = {
        "schema": SCHEMA,
        "approved_preregistration_digest": preregistration_digest(),
        "parent_marker_sha": PARENT_MARKER_SHA,
        "parent_run_id": PARENT_RUN_ID,
        "parent_final_artifact_id": PARENT_FINAL_ARTIFACT_ID,
        "parent_final_artifact_zip_digest": PARENT_FINAL_ARTIFACT_ZIP_DIGEST,
        "parent_final_json_sha256": PARENT_FINAL_JSON_SHA256,
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
        "group_ids": tuple(GROUP_IDS),
        "exposure_checkpoints": tuple(EXPOSURE_CHECKPOINTS),
        "exposures_per_world": EXPOSURES_PER_WORLD,
        "device": "cpu",
    }
    provisional = Exp326ExecutionIdentity(**values, exp326_execution_digest="")
    return Exp326ExecutionIdentity(**values, exp326_execution_digest=execution_digest(provisional))


def canonical_identity_json_bytes(identity: Exp326ExecutionIdentity) -> bytes:
    validate_execution_identity(identity)
    return canonical_json_bytes(asdict(identity)) + b"\n"
