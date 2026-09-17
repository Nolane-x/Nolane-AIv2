from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from .exp319_identity import (
    Exp319ExecutionIdentity,
    build_exp319_execution_identity,
    canonical_exp319_identity_json_bytes,
)

EXP319_MARKER_PATHS = (
    "protocols/v017/exp319_execution_identity_v1.json",
    "protocols/v017/exp319_execution_identity_v1.sha256",
)
EXP319_WORKFLOW_PATH = ".github/workflows/exp319-learnability-foundation-diagnostic.yml"
EXP319_CONTRACT_WORKFLOW_PATH = ".github/workflows/v017-exp319-contract.yml"
EXP319_COMPONENT_PATHS = {
    "generator": ("src/nolane_ai/experiments/exp319_worlds.py",),
    "training": (
        "src/nolane_ai/experiments/exp319_contract.py",
        "src/nolane_ai/experiments/exp319_metrics.py",
        "src/nolane_ai/experiments/exp319_training.py",
    ),
    "selection": ("src/nolane_ai/experiments/exp319_selection.py",),
    "scoring": (
        "src/nolane_ai/experiments/exp319_challenge.py",
        "src/nolane_ai/experiments/exp319_evidence.py",
    ),
}

EXP319_FROZEN_PATHS = (
    "docs/superpowers/specs/2026-09-16-exp319-learnability-foundation-diagnostic-design.md",
    "docs/superpowers/plans/2026-09-16-exp319-learnability-foundation-diagnostic.md",
    "protocols/v017/exp319_preregistration_v1.json",
    "protocols/v017/exp319_preregistration_v1.sha256",
    "src/nolane_ai/experiments/exp319_contract.py",
    "src/nolane_ai/experiments/exp319_worlds.py",
    "src/nolane_ai/experiments/exp319_metrics.py",
    "src/nolane_ai/experiments/exp319_training.py",
    "src/nolane_ai/experiments/exp319_selection.py",
    "src/nolane_ai/experiments/exp319_challenge.py",
    "src/nolane_ai/experiments/exp319_evidence.py",
    "src/nolane_ai/experiments/exp319_identity.py",
    "src/nolane_ai/experiments/exp319_freeze.py",
    "scripts/verify_exp319_contract.py",
    "scripts/exp319_stage_a_chunk.py",
    "scripts/exp319_stage_a_select.py",
    "scripts/exp319_stage_b_chunk.py",
    "scripts/exp319_stage_b_select.py",
    "scripts/exp319_materialize_heldout.py",
    "scripts/exp319_predict_heldout_shard.py",
    "scripts/exp319_seal_root.py",
    "scripts/exp319_finalize.py",
    "scripts/verify_exp319_freeze.py",
    EXP319_WORKFLOW_PATH,
    EXP319_CONTRACT_WORKFLOW_PATH,
    "tests/test_exp319_contract.py",
    "tests/test_exp319_worlds.py",
    "tests/test_exp319_metrics.py",
    "tests/test_exp319_training.py",
    "tests/test_exp319_selection.py",
    "tests/test_exp319_challenge.py",
    "tests/test_exp319_evidence.py",
    "tests/test_exp319_identity.py",
    "tests/test_exp319_freeze.py",
    "tests/test_exp319_cli.py",
    "tests/test_exp319_workflow.py",
    *EXP319_MARKER_PATHS,
)


def _digest_bytes(value: bytes, *, field: str) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{field} bytes must be non-empty bytes")
    return hashlib.sha256(value).hexdigest()


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


def _component_manifest_bytes(
    repo_root: Path,
    *,
    source_commit_sha: str,
    component: str,
    paths: tuple[str, ...],
) -> bytes:
    entries = [
        {
            "path": path,
            "blob_sha": _git(repo_root, "rev-parse", f"{source_commit_sha}:{path}"),
        }
        for path in paths
    ]
    payload = {
        "schema": "EXP319-FROZEN-COMPONENT-BLOBS-V1",
        "component": component,
        "files": entries,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_exp319_identity_from_components(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    generator_contract_bytes: bytes,
    training_contract_bytes: bytes,
    selection_contract_bytes: bytes,
    scoring_contract_bytes: bytes,
    workflow_bytes: bytes,
) -> Exp319ExecutionIdentity:
    return build_exp319_execution_identity(
        source_commit_sha=source_commit_sha,
        git_tree_sha=git_tree_sha,
        generator_contract_digest=_digest_bytes(generator_contract_bytes, field="generator contract"),
        training_contract_digest=_digest_bytes(training_contract_bytes, field="training contract"),
        selection_contract_digest=_digest_bytes(selection_contract_bytes, field="selection contract"),
        scoring_contract_digest=_digest_bytes(scoring_contract_bytes, field="scoring contract"),
        workflow_sha256=_digest_bytes(workflow_bytes, field="workflow"),
    )


def build_exp319_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> Exp319ExecutionIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    components = {
        name: _component_manifest_bytes(
            root,
            source_commit_sha=commit_sha,
            component=name,
            paths=paths,
        )
        for name, paths in EXP319_COMPONENT_PATHS.items()
    }
    workflow_bytes = _git_bytes(root, "show", f"{commit_sha}:{EXP319_WORKFLOW_PATH}")
    return build_exp319_identity_from_components(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        generator_contract_bytes=components["generator"],
        training_contract_bytes=components["training"],
        selection_contract_bytes=components["selection"],
        scoring_contract_bytes=components["scoring"],
        workflow_bytes=workflow_bytes,
    )


def canonical_marker_json_bytes(identity: Exp319ExecutionIdentity) -> bytes:
    return canonical_exp319_identity_json_bytes(identity)


def marker_sidecar_bytes(identity: Exp319ExecutionIdentity) -> bytes:
    payload = canonical_marker_json_bytes(identity)
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest}  {EXP319_MARKER_PATHS[0]}\n".encode("ascii")


def validate_frozen_changed_paths(paths: Iterable[str]) -> None:
    frozen = set(EXP319_FROZEN_PATHS)
    materialized = tuple(paths)
    forbidden = sorted({path for path in materialized if path not in frozen})
    if forbidden:
        raise ValueError(f"unrelated or forbidden EXP-319 changed paths: {forbidden}")


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if set(materialized) != set(EXP319_MARKER_PATHS) or len(materialized) != len(EXP319_MARKER_PATHS):
        raise ValueError("EXP-319 marker-only commit must change exactly the two execution identity files")
