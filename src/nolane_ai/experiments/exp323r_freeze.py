from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp323r_identity import (
    Exp323RExecutionIdentity,
    build_execution_identity,
    canonical_identity_json_bytes,
)


BASE_SHA = "45b539b5c69c982601ced9d642935472cc8458e1"
WORKFLOW_PATH = ".github/workflows/exp323r-second-decay-convergence.yml"
MARKER_PATHS = (
    "protocols/v017/exp323r_execution_identity_v1.json",
    "protocols/v017/exp323r_execution_identity_v1.sha256",
)
FROZEN_PATHS = (
    ".github/workflows/exp323r-reconstruction-materialization.yml",
    ".github/workflows/exp323r-second-decay-convergence.yml",
    ".github/workflows/v017-exp323r-repair.yml",
    "docs/superpowers/specs/2026-09-18-exp323r-materialized-reconstruction-repair.md",
    "docs/superpowers/specs/2026-09-18-exp323r-selection-lock.md",
    "protocols/v017/exp323r_reconstruction_selection_lock_v1.json",
    "protocols/v017/exp323r_reconstruction_selection_lock_v1.sha256",
    "scripts/exp323r_materialize_reconstruction.py",
    "scripts/exp323r_reduce.py",
    "scripts/exp323r_run_arm.py",
    "scripts/verify_exp323r_freeze.py",
    "src/nolane_ai/experiments/exp323r_identity.py",
    "src/nolane_ai/experiments/exp323r_reconstruction.py",
    "src/nolane_ai/experiments/exp323r_repair.py",
    "src/nolane_ai/experiments/exp323r_freeze.py",
    "tests/test_exp323r_identity.py",
    "tests/test_exp323r_locked_repair.py",
    "tests/test_exp323r_reconstruction.py",
    "tests/test_exp323r_freeze.py",
    *MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git","-C",str(root),*args])


def build_identity_from_git(repo_root: str | Path, source_commit_sha: str = "HEAD") -> Exp323RExecutionIdentity:
    root=Path(repo_root).resolve()
    commit=_git(root,"rev-parse",source_commit_sha)
    tree=_git(root,"rev-parse",f"{commit}^{{tree}}")
    workflow=_git_bytes(root,"show",f"{commit}:{WORKFLOW_PATH}")
    return build_execution_identity(
        source_commit_sha=commit,
        git_tree_sha=tree,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def marker_json_bytes(identity: Exp323RExecutionIdentity) -> bytes:
    return canonical_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp323RExecutionIdentity) -> bytes:
    payload=marker_json_bytes(identity)
    digest=hashlib.sha256(payload).hexdigest()
    return f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")


def validate_source_paths(paths: Iterable[str]) -> None:
    bad=sorted(set(paths)-set(FROZEN_PATHS))
    if bad:
        raise ValueError(f"unrelated EXP-323R changed paths: {bad}")


def validate_marker_paths(paths: Iterable[str]) -> None:
    materialized=tuple(paths)
    if len(materialized)!=2 or set(materialized)!=set(MARKER_PATHS):
        raise ValueError("EXP-323R marker-only commit must change exactly two identity files")
