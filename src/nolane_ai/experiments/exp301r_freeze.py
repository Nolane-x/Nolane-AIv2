from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp301r_identity import (
    RecoveryExecutionIdentity,
    build_recovery_execution_identity,
    canonical_recovery_identity_json_bytes,
)

EXP301R_MARKER_PATHS = (
    "protocols/v017/exp301r_execution_identity_v1.json",
    "protocols/v017/exp301r_execution_identity_v1.sha256",
)
EXP301R_WORKFLOW_PATH = ".github/workflows/exp301r-standard-runner-sharded-recovery.yml"


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def build_recovery_identity_from_components(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    workflow_bytes: bytes,
) -> RecoveryExecutionIdentity:
    if not isinstance(workflow_bytes, bytes) or not workflow_bytes:
        raise ValueError("workflow bytes must be non-empty bytes")
    return build_recovery_execution_identity(
        source_commit_sha=source_commit_sha,
        git_tree_sha=git_tree_sha,
        workflow_sha256=hashlib.sha256(workflow_bytes).hexdigest(),
    )


def build_recovery_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> RecoveryExecutionIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    workflow_bytes = _git_bytes(root, "show", f"{commit_sha}:{EXP301R_WORKFLOW_PATH}")
    return build_recovery_identity_from_components(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        workflow_bytes=workflow_bytes,
    )


def canonical_marker_json_bytes(identity: RecoveryExecutionIdentity) -> bytes:
    return canonical_recovery_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: RecoveryExecutionIdentity) -> bytes:
    payload = canonical_marker_json_bytes(identity)
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {EXP301R_MARKER_PATHS[0]}\n".encode("ascii")


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if set(materialized) != set(EXP301R_MARKER_PATHS) or len(materialized) != len(EXP301R_MARKER_PATHS):
        raise ValueError("EXP-301R marker-only commit must change exactly the two recovery execution identity files")
