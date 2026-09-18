from __future__ import annotations

import hashlib

import pytest

from nolane_ai.experiments.exp321_freeze import (
    EXP321_FROZEN_PATHS,
    EXP321_MARKER_PATHS,
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_frozen_changed_paths,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp321_identity import build_exp321_execution_identity


def _h(ch: str, n: int) -> str:
    return ch * n


def _identity():
    return build_exp321_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )


def test_frozen_allowlist_contains_complete_exp321_surface() -> None:
    required = {
        "src/nolane_ai/experiments/exp321_contract.py",
        "src/nolane_ai/experiments/exp321_localization.py",
        "src/nolane_ai/experiments/exp321_measure.py",
        "src/nolane_ai/experiments/exp321_identity.py",
        "src/nolane_ai/experiments/exp321_freeze.py",
        "scripts/exp321_localize_checkpoint.py",
        "scripts/verify_exp321_contract.py",
        "scripts/verify_exp321_freeze.py",
        ".github/workflows/exp321-afixed-failure-localization.yml",
        ".github/workflows/v017-exp321-contract.yml",
        *EXP321_MARKER_PATHS,
    }
    assert required <= set(EXP321_FROZEN_PATHS)
    validate_frozen_changed_paths(required)
    with pytest.raises(ValueError, match="unrelated|forbidden"):
        validate_frozen_changed_paths((*required, "README.md"))


def test_marker_json_and_sidecar_are_reproducible() -> None:
    identity = _identity()
    payload = canonical_marker_json_bytes(identity)
    assert payload == canonical_marker_json_bytes(identity)
    expected = hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(identity) == (
        f"{expected}  {EXP321_MARKER_PATHS[0]}\n".encode("ascii")
    )


def test_marker_commit_is_exactly_two_identity_files() -> None:
    validate_marker_changed_paths(EXP321_MARKER_PATHS)
    validate_marker_changed_paths(tuple(reversed(EXP321_MARKER_PATHS)))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((EXP321_MARKER_PATHS[0],))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((*EXP321_MARKER_PATHS, "README.md"))
