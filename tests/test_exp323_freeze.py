from __future__ import annotations

import hashlib

import pytest

from nolane_ai.experiments.exp323_freeze import (
    EXP323_BASE_SHA,
    EXP323_FROZEN_PATHS,
    EXP323_MARKER_PATHS,
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_frozen_changed_paths,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp323_identity import build_exp323_execution_identity


def _identity():
    return build_exp323_execution_identity(
        source_commit_sha="a" * 40,
        git_tree_sha="b" * 40,
        workflow_sha256="c" * 64,
    )


def test_freeze_is_bound_to_exact_exp322_marker_parent() -> None:
    assert EXP323_BASE_SHA == "4a7f079a97f2efeeb4682abe05246e893d30c4ad"


def test_frozen_allowlist_contains_complete_exp323_surface() -> None:
    required = {
        "src/nolane_ai/experiments/exp323_contract.py",
        "src/nolane_ai/experiments/exp323_evidence.py",
        "src/nolane_ai/experiments/exp323_runtime.py",
        "src/nolane_ai/experiments/exp323_identity.py",
        "src/nolane_ai/experiments/exp323_freeze.py",
        "scripts/exp323_run_arm.py",
        "scripts/exp323_reduce.py",
        "scripts/verify_exp323_contract.py",
        "scripts/verify_exp323_freeze.py",
        ".github/workflows/exp323-second-decay-convergence.yml",
        ".github/workflows/v017-exp323-contract.yml",
        *EXP323_MARKER_PATHS,
    }
    assert required <= set(EXP323_FROZEN_PATHS)
    validate_frozen_changed_paths(required)
    with pytest.raises(ValueError, match="unrelated|forbidden"):
        validate_frozen_changed_paths((*required, "README.md"))


def test_marker_bytes_are_reproducible() -> None:
    identity = _identity()
    payload = canonical_marker_json_bytes(identity)
    expected = hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(identity) == (
        f"{expected}  {EXP323_MARKER_PATHS[0]}\n".encode("ascii")
    )


def test_marker_commit_is_exactly_two_identity_files() -> None:
    validate_marker_changed_paths(EXP323_MARKER_PATHS)
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((EXP323_MARKER_PATHS[0],))
