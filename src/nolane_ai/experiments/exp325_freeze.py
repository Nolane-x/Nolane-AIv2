from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp325_identity import (
    Exp325ExecutionIdentity,
    build_execution_identity,
    canonical_identity_json_bytes,
)


BASE_SHA = "3bc060f1d3656dbae1de2265a32b145d9cff0dbc"
WORKFLOW_PATH = ".github/workflows/exp325-single-world-memorization.yml"
MARKER_PATHS = (
    "protocols/v017/exp325_execution_identity_v1.json",
    "protocols/v017/exp325_execution_identity_v1.sha256",
)
FROZEN_PATHS = (
    ".github/workflows/exp325-single-world-memorization.yml",
    ".github/workflows/v017-exp325-contract.yml",
    "docs/superpowers/specs/2026-09-18-exp325-single-world-memorization-design.md",
    "protocols/v017/exp325_preregistration_v1.json",
    "protocols/v017/exp325_preregistration_v1.sha256",
    "scripts/exp325_reduce.py",
    "scripts/exp325_run_family.py",
    "scripts/verify_exp325_contract.py",
    "scripts/verify_exp325_freeze.py",
    "src/nolane_ai/experiments/exp325_contract.py",
    "src/nolane_ai/experiments/exp325_identity.py",
    "src/nolane_ai/experiments/exp325_runtime.py",
    "src/nolane_ai/experiments/exp325_freeze.py",
    "tests/test_exp325_contract.py",
    "tests/test_exp325_identity.py",
    "tests/test_exp325_runtime.py",
    "tests/test_exp325_workflow.py",
    "tests/test_exp325_freeze.py",
    *MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git","-C",str(root),*args])


def build_identity_from_git(repo_root: str | Path, source_commit_sha: str = "HEAD") -> Exp325ExecutionIdentity:
    root=Path(repo_root).resolve()
    commit=_git(root,"rev-parse",source_commit_sha)
    tree=_git(root,"rev-parse",f"{commit}^{{tree}}")
    workflow=_git_bytes(root,"show",f"{commit}:{WORKFLOW_PATH}")
    return build_execution_identity(
        source_commit_sha=commit,
        git_tree_sha=tree,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def marker_json_bytes(identity: Exp325ExecutionIdentity) -> bytes:
    return canonical_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp325ExecutionIdentity) -> bytes:
    payload=marker_json_bytes(identity)
    digest=hashlib.sha256(payload).hexdigest()
    return f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")


def validate_source_paths(paths: Iterable[str]) -> None:
    bad=sorted(set(paths)-set(FROZEN_PATHS))
    if bad:
        raise ValueError(f"unrelated EXP-325 changed paths: {bad}")


def validate_marker_paths(paths: Iterable[str]) -> None:
    materialized=tuple(paths)
    if len(materialized)!=2 or set(materialized)!=set(MARKER_PATHS):
        raise ValueError("EXP-325 marker-only commit must change exactly two identity files")
