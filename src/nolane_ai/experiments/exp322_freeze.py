from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp322_identity import (
    Exp322ExecutionIdentity,
    build_exp322_execution_identity,
    canonical_exp322_identity_json_bytes,
)

EXP322_BASE_SHA = "a2f1004206aeffb37c327fcd909f0c1564e67045"
EXP322_WORKFLOW_PATH = ".github/workflows/exp322-teacher-forced-learnability.yml"
EXP322_MARKER_PATHS = (
    "protocols/v017/exp322_execution_identity_v1.json",
    "protocols/v017/exp322_execution_identity_v1.sha256",
)
EXP322_FROZEN_PATHS = (
    "docs/superpowers/specs/2026-09-18-exp322-teacher-forced-learnability-intervention-design.md",
    "protocols/v017/exp322_preregistration_v1.json",
    "protocols/v017/exp322_preregistration_v1.sha256",
    "src/nolane_ai/experiments/exp322_contract.py",
    "src/nolane_ai/experiments/exp322_evidence.py",
    "src/nolane_ai/experiments/exp322_runtime.py",
    "src/nolane_ai/experiments/exp322_identity.py",
    "src/nolane_ai/experiments/exp322_freeze.py",
    "scripts/verify_exp322_contract.py",
    "scripts/verify_exp322_freeze.py",
    "scripts/exp322_run_arm.py",
    "scripts/exp322_reduce.py",
    ".github/workflows/v017-exp322-contract.yml",
    EXP322_WORKFLOW_PATH,
    "tests/test_exp322_contract.py",
    "tests/test_exp322_evidence.py",
    "tests/test_exp322_runtime.py",
    "tests/test_exp322_identity.py",
    "tests/test_exp322_freeze.py",
    "tests/test_exp322_workflow.py",
    *EXP322_MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def build_exp322_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> Exp322ExecutionIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    workflow = _git_bytes(root, "show", f"{commit_sha}:{EXP322_WORKFLOW_PATH}")
    return build_exp322_execution_identity(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def canonical_marker_json_bytes(identity: Exp322ExecutionIdentity) -> bytes:
    return canonical_exp322_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp322ExecutionIdentity) -> bytes:
    payload = canonical_marker_json_bytes(identity)
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {EXP322_MARKER_PATHS[0]}\n".encode("ascii")


def validate_frozen_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    frozen = set(EXP322_FROZEN_PATHS)
    forbidden = sorted({path for path in materialized if path not in frozen})
    if forbidden:
        raise ValueError(f"unrelated or forbidden EXP-322 changed paths: {forbidden}")


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if set(materialized) != set(EXP322_MARKER_PATHS) or len(materialized) != len(EXP322_MARKER_PATHS):
        raise ValueError(
            "EXP-322 marker-only commit must change exactly the two execution identity files"
        )
