from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp323_identity import (
    Exp323ExecutionIdentity,
    build_exp323_execution_identity,
    canonical_exp323_identity_json_bytes,
)


EXP323_BASE_SHA = "4a7f079a97f2efeeb4682abe05246e893d30c4ad"
EXP323_WORKFLOW_PATH = ".github/workflows/exp323-second-decay-convergence.yml"
EXP323_MARKER_PATHS = (
    "protocols/v017/exp323_execution_identity_v1.json",
    "protocols/v017/exp323_execution_identity_v1.sha256",
)
EXP323_FROZEN_PATHS = (
    "docs/superpowers/specs/2026-09-18-exp323-second-decay-convergence-intervention-design.md",
    "protocols/v017/exp323_preregistration_v1.json",
    "protocols/v017/exp323_preregistration_v1.sha256",
    "src/nolane_ai/experiments/exp323_contract.py",
    "src/nolane_ai/experiments/exp323_evidence.py",
    "src/nolane_ai/experiments/exp323_runtime.py",
    "src/nolane_ai/experiments/exp323_identity.py",
    "src/nolane_ai/experiments/exp323_freeze.py",
    "scripts/verify_exp323_contract.py",
    "scripts/verify_exp323_freeze.py",
    "scripts/verify_exp323_replay.py",
    "scripts/exp323_run_arm.py",
    "scripts/exp323_reduce.py",
    ".github/workflows/v017-exp323-contract.yml",
    ".github/workflows/v017-exp323-replay-authority.yml",
    EXP323_WORKFLOW_PATH,
    "tests/test_exp323_contract.py",
    "tests/test_exp323_evidence.py",
    "tests/test_exp323_runtime.py",
    "tests/test_exp323_identity.py",
    "tests/test_exp323_freeze.py",
    "tests/test_exp323_replay_workflow.py",
    "tests/test_exp323_workflow.py",
    *EXP323_MARKER_PATHS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def build_exp323_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> Exp323ExecutionIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    workflow = _git_bytes(root, "show", f"{commit_sha}:{EXP323_WORKFLOW_PATH}")
    return build_exp323_execution_identity(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        workflow_sha256=hashlib.sha256(workflow).hexdigest(),
    )


def canonical_marker_json_bytes(identity: Exp323ExecutionIdentity) -> bytes:
    return canonical_exp323_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp323ExecutionIdentity) -> bytes:
    payload = canonical_marker_json_bytes(identity)
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {EXP323_MARKER_PATHS[0]}\n".encode("ascii")


def validate_frozen_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    frozen = set(EXP323_FROZEN_PATHS)
    forbidden = sorted({path for path in materialized if path not in frozen})
    if forbidden:
        raise ValueError(f"unrelated or forbidden EXP-323 changed paths: {forbidden}")


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if (
        set(materialized) != set(EXP323_MARKER_PATHS)
        or len(materialized) != len(EXP323_MARKER_PATHS)
    ):
        raise ValueError(
            "EXP-323 marker-only commit must change exactly the two execution identity files"
        )
