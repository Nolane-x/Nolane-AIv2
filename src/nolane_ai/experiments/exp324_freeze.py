from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp324_identity import (
    Exp324ExecutionIdentity,
    build_execution_identity,
    canonical_identity_json_bytes,
)


BASE_SHA = "217477a911aabbb764741e76cca38e54013e7da6"
WORKFLOW_PATH = ".github/workflows/exp324-family-isolated-learnability.yml"
MARKER_PATHS = (
    "protocols/v017/exp324_execution_identity_v1.json",
    "protocols/v017/exp324_execution_identity_v1.sha256",
)
FROZEN_PATHS = (
    ".github/workflows/exp324-family-isolated-learnability.yml",
    ".github/workflows/v017-exp324-contract.yml",
    ".github/workflows/v017-exp324-runtime.yml",
    "docs/superpowers/specs/2026-09-18-exp324-family-isolated-learnability-design.md",
    "protocols/v017/exp324_preregistration_v1.json",
    "protocols/v017/exp324_preregistration_v1.sha256",
    "scripts/exp324_reduce.py",
    "scripts/exp324_run_family.py",
    "scripts/verify_exp324_contract.py",
    "scripts/verify_exp324_freeze.py",
    "src/nolane_ai/experiments/exp324_contract.py",
    "src/nolane_ai/experiments/exp324_identity.py",
    "src/nolane_ai/experiments/exp324_runtime.py",
    "src/nolane_ai/experiments/exp324_freeze.py",
    "tests/test_exp324_contract.py",
    "tests/test_exp324_evidence.py",
    "tests/test_exp324_identity.py",
    "tests/test_exp324_runtime.py",
    "tests/test_exp324_freeze.py",
    "tests/test_exp324_workflow.py",
    *MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git","-C",str(root),*args])


def build_identity_from_git(repo_root: str | Path, source_commit_sha: str = "HEAD") -> Exp324ExecutionIdentity:
    root=Path(repo_root).resolve()
    commit=_git(root,"rev-parse",source_commit_sha)
    tree=_git(root,"rev-parse",f"{commit}^{{tree}}")
    workflow=_git_bytes(root,"show",f"{commit}:{WORKFLOW_PATH}")
    return build_execution_identity(
        source_commit_sha=commit,
        git_tree_sha=tree,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def marker_json_bytes(identity: Exp324ExecutionIdentity) -> bytes:
    return canonical_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp324ExecutionIdentity) -> bytes:
    payload=marker_json_bytes(identity)
    digest=hashlib.sha256(payload).hexdigest()
    return f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")


def validate_source_paths(paths: Iterable[str]) -> None:
    bad=sorted(set(paths)-set(FROZEN_PATHS))
    if bad:
        raise ValueError(f"unrelated EXP-324 changed paths: {bad}")


def validate_marker_paths(paths: Iterable[str]) -> None:
    materialized=tuple(paths)
    if len(materialized)!=2 or set(materialized)!=set(MARKER_PATHS):
        raise ValueError("EXP-324 marker-only commit must change exactly two identity files")
