from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp326_identity import (
    Exp326ExecutionIdentity,
    build_execution_identity,
    canonical_identity_json_bytes,
)


BASE_SHA = "b24830c842258258f42ffb92e5f00277f8db8e9b"
WORKFLOW_PATH = ".github/workflows/exp326-balanced-multiworld-breakpoint.yml"
MARKER_PATHS = (
    "protocols/v017/exp326_execution_identity_v1.json",
    "protocols/v017/exp326_execution_identity_v1.sha256",
)
FROZEN_PATHS = (
    ".github/workflows/exp326-balanced-multiworld-breakpoint.yml",
    ".github/workflows/v017-exp326-contract.yml",
    "docs/superpowers/specs/2026-09-19-exp326-balanced-multiworld-breakpoint.md",
    "protocols/v017/exp326_preregistration_v1.json",
    "protocols/v017/exp326_preregistration_v1.sha256",
    "scripts/exp326_reduce.py",
    "scripts/exp326_run_family.py",
    "scripts/verify_exp326_contract.py",
    "scripts/verify_exp326_freeze.py",
    "src/nolane_ai/experiments/exp326_contract.py",
    "src/nolane_ai/experiments/exp326_identity.py",
    "src/nolane_ai/experiments/exp326_runtime.py",
    "src/nolane_ai/experiments/exp326_freeze.py",
    "tests/test_exp326_contract.py",
    "tests/test_exp326_identity.py",
    "tests/test_exp326_runtime.py",
    "tests/test_exp326_workflow.py",
    "tests/test_exp326_freeze.py",
    *MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git","-C",str(root),*args], text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git","-C",str(root),*args])


def build_identity_from_git(repo_root: str | Path, source_commit_sha: str = "HEAD") -> Exp326ExecutionIdentity:
    root=Path(repo_root).resolve()
    commit=_git(root,"rev-parse",source_commit_sha)
    tree=_git(root,"rev-parse",f"{commit}^{{tree}}")
    workflow=_git_bytes(root,"show",f"{commit}:{WORKFLOW_PATH}")
    return build_execution_identity(
        source_commit_sha=commit,
        git_tree_sha=tree,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def marker_json_bytes(identity: Exp326ExecutionIdentity) -> bytes:
    return canonical_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp326ExecutionIdentity) -> bytes:
    payload=marker_json_bytes(identity)
    digest=hashlib.sha256(payload).hexdigest()
    return f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")


def validate_source_paths(paths: Iterable[str]) -> None:
    bad=sorted(set(paths)-set(FROZEN_PATHS))
    if bad:
        raise ValueError(f"unrelated EXP-326 changed paths: {bad}")


def validate_marker_paths(paths: Iterable[str]) -> None:
    materialized=tuple(paths)
    if len(materialized)!=2 or set(materialized)!=set(MARKER_PATHS):
        raise ValueError("EXP-326 marker-only commit must change exactly two identity files")
