from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .exp301r_recovery import (
    FROZEN_IMPLEMENTATION_DIGEST,
    PRIOR_FAILED_RUN_ID,
    RECOVERY_SCHEMA,
    SHARD_COUNT,
    WORLDS_PER_SHARD,
)

EXP301R_IDENTITY_SCHEMA = "EXP301R-RECOVERY-EXECUTION-IDENTITY-V1"
ORIGINAL_EXP301_MARKER_SHA = "bac51c29c46e4c1fb3db5445a299da4674fdb6d8"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RecoveryExecutionIdentity:
    schema: str
    recovery_schema: str
    original_exp301_marker_sha: str
    frozen_implementation_digest: str
    prior_failed_run_id: int
    source_commit_sha: str
    source_tree_digest: str
    workflow_sha256: str
    shard_count: int
    worlds_per_shard: int
    device: str
    recovery_execution_digest: str


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _require_hex(value: str, *, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


def source_tree_digest_from_git_tree_sha(git_tree_sha: str) -> str:
    _require_hex(git_tree_sha, length=40, field="git tree")
    return hashlib.sha256(f"EXP301R-GIT-TREE-V1|{git_tree_sha}".encode("ascii")).hexdigest()


def canonical_recovery_execution_digest(identity: RecoveryExecutionIdentity) -> str:
    payload = asdict(identity)
    payload.pop("recovery_execution_digest", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_recovery_execution_identity(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    workflow_sha256: str,
) -> RecoveryExecutionIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(workflow_sha256, length=64, field="workflow sha256")
    source_tree_digest = source_tree_digest_from_git_tree_sha(git_tree_sha)
    values = {
        "schema": EXP301R_IDENTITY_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "original_exp301_marker_sha": ORIGINAL_EXP301_MARKER_SHA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest,
        "workflow_sha256": workflow_sha256,
        "shard_count": SHARD_COUNT,
        "worlds_per_shard": WORLDS_PER_SHARD,
        "device": "cpu",
    }
    provisional = RecoveryExecutionIdentity(**values, recovery_execution_digest="")
    return RecoveryExecutionIdentity(
        **values,
        recovery_execution_digest=canonical_recovery_execution_digest(provisional),
    )


def canonical_recovery_identity_json_bytes(identity: RecoveryExecutionIdentity) -> bytes:
    if identity.recovery_execution_digest != canonical_recovery_execution_digest(identity):
        raise ValueError("recovery execution digest mismatch")
    return _canonical_bytes(asdict(identity)) + b"\n"
