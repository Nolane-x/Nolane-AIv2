from __future__ import annotations

import hashlib

import pytest

from nolane_ai.experiments.exp322_freeze import (
    EXP322_BASE_SHA,
    EXP322_FROZEN_PATHS,
    EXP322_MARKER_PATHS,
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_frozen_changed_paths,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp322_identity import build_exp322_execution_identity


def _identity():
    return build_exp322_execution_identity(
        source_commit_sha="a" * 40,
        git_tree_sha="b" * 40,
        workflow_sha256="c" * 64,
    )


def test_freeze_is_bound_to_exact_exp321_marker_parent() -> None:
    assert EXP322_BASE_SHA == "a2f1004206aeffb37c327fcd909f0c1564e67045"


def test_frozen_allowlist_contains_complete_exp322_surface() -> None:
    required = {
        "src/nolane_ai/experiments/exp322_contract.py",
        "src/nolane_ai/experiments/exp322_evidence.py",
        "src/nolane_ai/experiments/exp322_runtime.py",
        "src/nolane_ai/experiments/exp322_identity.py",
        "src/nolane_ai/experiments/exp322_freeze.py",
        "scripts/exp322_run_arm.py",
        "scripts/exp322_reduce.py",
        "scripts/verify_exp322_contract.py",
        "scripts/verify_exp322_freeze.py",
        ".github/workflows/exp322-teacher-forced-learnability.yml",
        ".github/workflows/v017-exp322-contract.yml",
        *EXP322_MARKER_PATHS,
    }
    assert required <= set(EXP322_FROZEN_PATHS)
    validate_frozen_changed_paths(required)
    with pytest.raises(ValueError, match="unrelated|forbidden"):
        validate_frozen_changed_paths((*required, "README.md"))


def test_marker_bytes_are_reproducible() -> None:
    identity = _identity()
    payload = canonical_marker_json_bytes(identity)
    expected = hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(identity) == (
        f"{expected}  {EXP322_MARKER_PATHS[0]}\n".encode("ascii")
    )


def test_marker_commit_is_exactly_two_identity_files() -> None:
    validate_marker_changed_paths(EXP322_MARKER_PATHS)
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((EXP322_MARKER_PATHS[0],))
