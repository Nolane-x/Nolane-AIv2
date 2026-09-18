from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Iterable

from .exp321_identity import (
    Exp321ExecutionIdentity,
    build_exp321_execution_identity,
    canonical_exp321_identity_json_bytes,
)


EXP321_MARKER_PATHS = (
    "protocols/v017/exp321_execution_identity_v1.json",
    "protocols/v017/exp321_execution_identity_v1.sha256",
)
EXP321_WORKFLOW_PATH = ".github/workflows/exp321-afixed-failure-localization.yml"
EXP321_FROZEN_PATHS = (
    "docs/superpowers/specs/2026-09-18-exp321-afixed-stage-a-failure-localization-design.md",
    "protocols/v017/exp321_preregistration_v1.json",
    "protocols/v017/exp321_preregistration_v1.sha256",
    "src/nolane_ai/experiments/exp321_contract.py",
    "src/nolane_ai/experiments/exp321_localization.py",
    "src/nolane_ai/experiments/exp321_measure.py",
    "src/nolane_ai/experiments/exp321_identity.py",
    "src/nolane_ai/experiments/exp321_freeze.py",
    "scripts/exp321_localize_checkpoint.py",
    "scripts/verify_exp321_contract.py",
    "scripts/verify_exp321_freeze.py",
    EXP321_WORKFLOW_PATH,
    ".github/workflows/v017-exp321-contract.yml",
    "tests/test_exp321_contract.py",
    "tests/test_exp321_measurement_surface.py",
    "tests/test_exp321_identity.py",
    "tests/test_exp321_freeze.py",
    "tests/test_exp321_workflow.py",
    *EXP321_MARKER_PATHS,
)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), *args])


def build_exp321_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> Exp321ExecutionIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    workflow_bytes = _git_bytes(
        root,
        "show",
        f"{commit_sha}:{EXP321_WORKFLOW_PATH}",
    )
    return build_exp321_execution_identity(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        workflow_sha256=hashlib.sha256(workflow_bytes).hexdigest(),
    )


def canonical_marker_json_bytes(identity: Exp321ExecutionIdentity) -> bytes:
    return canonical_exp321_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp321ExecutionIdentity) -> bytes:
    payload = canonical_marker_json_bytes(identity)
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {EXP321_MARKER_PATHS[0]}\n".encode("ascii")


def validate_frozen_changed_paths(paths: Iterable[str]) -> None:
    frozen = set(EXP321_FROZEN_PATHS)
    materialized = tuple(paths)
    forbidden = sorted({path for path in materialized if path not in frozen})
    if forbidden:
        raise ValueError(f"unrelated or forbidden EXP-321 changed paths: {forbidden}")


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if (
        set(materialized) != set(EXP321_MARKER_PATHS)
        or len(materialized) != len(EXP321_MARKER_PATHS)
    ):
        raise ValueError(
            "EXP-321 marker-only commit must change exactly the two execution identity files"
        )
