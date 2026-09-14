from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from .exp301_compute import COMPUTE_LEDGER_VERSION
from .exp301_execution import scientific_execution_contract_digest
from .exp301_identity import (
    EXP301_PREREG_V2_DIGEST,
    FrozenImplementationIdentity,
    build_frozen_implementation_identity,
)


EXP301_MARKER_PATHS = (
    "protocols/v017/exp301_execution_identity_v1.json",
    "protocols/v017/exp301_execution_identity_v1.sha256",
)
EXP301_WORLD_PATH = "src/nolane_ai/experiments/exp301_worlds.py"
EXP301_ANALYSIS_PATHS = (
    "src/nolane_ai/experiments/exp301_analysis.py",
    "src/nolane_ai/experiments/exp301_cross_root.py",
    "src/nolane_ai/experiments/exp301_evidence.py",
    "src/nolane_ai/experiments/exp301_evaluation.py",
    "src/nolane_ai/experiments/exp301_compute.py",
    "src/nolane_ai/experiments/exp301_execution.py",
    "src/nolane_ai/experiments/exp301_identity.py",
)
EXP301_WORKFLOW_PATH = ".github/workflows/exp301-scientific-court.yml"
EXP301_ROOTS = (0, 1, 2, 3)
EXP301_TASK_FAMILIES = (
    "iterative-grid-and-maze",
    "algorithmic-sequence-transform",
    "generator-heldout-abstract-transformation",
    "language-sequence-control",
)
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hex(value: str, *, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")


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


def source_tree_digest_from_git_tree_sha(git_tree_sha: str) -> str:
    _require_hex(git_tree_sha, length=40, field="git tree")
    return hashlib.sha256(f"EXP301-GIT-TREE-V1|{git_tree_sha}".encode("ascii")).hexdigest()


def parameter_audit_payload() -> dict[str, object]:
    return {
        "schema": "EXP301-PARAMETER-AUDIT-CONSTITUTION-V1",
        "total_budget": 10_000_000,
        "frozen_base": 9_120_832,
        "capacity_exchange_envelope": 879_168,
        "arm_trainable_parameters": {
            "A_FIXED": 10_000_000,
            "B_LOOP_SIMPLE": 10_000_000,
            "C_NRS_CORE": 10_000_000,
        },
        "dead_capacity_forbidden": True,
    }


def parameter_audit_digest() -> str:
    return _digest(parameter_audit_payload())


def architecture_receipts_payload() -> dict[str, object]:
    return {
        "schema": "EXP301-ARCHITECTURE-RECEIPTS-CONSTITUTION-V2",
        "vocab_size": 4_608,
        "d_model": 448,
        "n_heads": 7,
        "head_dim": 64,
        "shared_layers": 3,
        "d_ff": 1_152,
        "effort_multipliers": (1, 2, 4, 8, 12, 16),
        "arms": {
            "A_FIXED": {
                "capacity_bottleneck": 981,
                "capacity_tail_parameters": 192,
                "weight_tied_across_depth": False,
                "loop_conditioning": False,
                "stateless_restarts": True,
                "latent_state_carry": False,
                "trainable_parameters": 10_000_000,
            },
            "B_LOOP_SIMPLE": {
                "capacity_bottleneck": 981,
                "capacity_tail_parameters": 192,
                "weight_tied_across_depth": True,
                "loop_conditioning": False,
                "stateless_restarts": False,
                "latent_state_carry": True,
                "trainable_parameters": 10_000_000,
            },
            "C_NRS_CORE": {
                "capacity_bottleneck": 973,
                "capacity_tail_parameters": 192,
                "weight_tied_across_depth": True,
                "loop_conditioning": True,
                "stateless_restarts": False,
                "latent_state_carry": True,
                "trainable_parameters": 10_000_000,
            },
        },
    }


def architecture_receipts_digest() -> str:
    return _digest(architecture_receipts_payload())


def generator_contract_digest(*, source_blob_sha: str, split: str, per_family: int) -> str:
    _require_hex(source_blob_sha, length=40, field="generator source blob")
    if split not in {"train", "development", "challenge"}:
        raise ValueError("generator split must be train, development, or challenge")
    if not isinstance(per_family, int) or per_family <= 0:
        raise ValueError("per_family must be a positive integer")
    payload = {
        "schema": "EXP301-GENERATOR-CONTRACT-V1",
        "source_blob_sha": source_blob_sha,
        "split": split,
        "per_family": per_family,
        "roots": EXP301_ROOTS,
        "families": EXP301_TASK_FAMILIES,
        "challenge_policy": (
            "post-freeze-beacon-required; no challenge nonce or materialization in freeze"
            if split == "challenge"
            else "deterministic pre-challenge split"
        ),
    }
    return _digest(payload)


def analysis_contract_digest(blob_shas: Iterable[str]) -> str:
    materialized = tuple(blob_shas)
    if not materialized:
        raise ValueError("analysis blob set must be non-empty")
    for sha in materialized:
        _require_hex(sha, length=40, field="analysis blob")
    return _digest(
        {
            "schema": "EXP301-ANALYSIS-CODE-BLOBS-V1",
            "blob_shas": tuple(sorted(materialized)),
        }
    )


def workflow_contract_digest(workflow_blob_sha: str) -> str:
    _require_hex(workflow_blob_sha, length=40, field="workflow blob")
    return _digest(
        {
            "schema": "EXP301-SCIENTIFIC-WORKFLOW-BLOB-V1",
            "blob_sha": workflow_blob_sha,
        }
    )


def build_freeze_identity_from_components(
    *,
    source_commit_sha: str,
    git_tree_sha: str,
    worlds_blob_sha: str,
    analysis_blob_shas: Iterable[str],
    workflow_blob_sha: str,
) -> FrozenImplementationIdentity:
    _require_hex(source_commit_sha, length=40, field="source commit")
    _require_hex(worlds_blob_sha, length=40, field="worlds blob")
    return build_frozen_implementation_identity(
        source_commit_sha=source_commit_sha,
        source_tree_digest=source_tree_digest_from_git_tree_sha(git_tree_sha),
        prereg_semantic_digest=EXP301_PREREG_V2_DIGEST,
        architecture_receipts_digest=architecture_receipts_digest(),
        scientific_execution_contract_digest=scientific_execution_contract_digest(),
        train_generator_digest=generator_contract_digest(
            source_blob_sha=worlds_blob_sha,
            split="train",
            per_family=128,
        ),
        development_generator_digest=generator_contract_digest(
            source_blob_sha=worlds_blob_sha,
            split="development",
            per_family=64,
        ),
        challenge_generator_digest=generator_contract_digest(
            source_blob_sha=worlds_blob_sha,
            split="challenge",
            per_family=128,
        ),
        parameter_audit_digest=parameter_audit_digest(),
        compute_ledger_version=COMPUTE_LEDGER_VERSION,
        analysis_digest=analysis_contract_digest(analysis_blob_shas),
        workflow_digest=workflow_contract_digest(workflow_blob_sha),
    )


def build_freeze_identity_from_git(
    repo_root: str | Path,
    source_commit_sha: str = "HEAD",
) -> FrozenImplementationIdentity:
    root = Path(repo_root).resolve()
    commit_sha = _git(root, "rev-parse", source_commit_sha)
    _require_hex(commit_sha, length=40, field="source commit")
    tree_sha = _git(root, "rev-parse", f"{commit_sha}^{{tree}}")
    worlds_blob = _git(root, "rev-parse", f"{commit_sha}:{EXP301_WORLD_PATH}")
    analysis_blobs = tuple(
        _git(root, "rev-parse", f"{commit_sha}:{path}")
        for path in EXP301_ANALYSIS_PATHS
    )
    workflow_blob = _git(root, "rev-parse", f"{commit_sha}:{EXP301_WORKFLOW_PATH}")
    return build_freeze_identity_from_components(
        source_commit_sha=commit_sha,
        git_tree_sha=tree_sha,
        worlds_blob_sha=worlds_blob,
        analysis_blob_shas=analysis_blobs,
        workflow_blob_sha=workflow_blob,
    )


def canonical_identity_json_bytes(identity: FrozenImplementationIdentity) -> bytes:
    return _canonical_bytes(asdict(identity)) + b"\n"


def validate_marker_changed_paths(paths: Iterable[str]) -> None:
    materialized = tuple(paths)
    if set(materialized) != set(EXP301_MARKER_PATHS) or len(materialized) != len(EXP301_MARKER_PATHS):
        raise ValueError(
            "EXP-301 marker-only commit must change exactly the two execution identity files"
        )
