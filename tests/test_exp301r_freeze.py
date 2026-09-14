import hashlib

import pytest

from nolane_ai.experiments.exp301r_freeze import (
    EXP301R_MARKER_PATHS,
    build_recovery_identity_from_components,
    canonical_marker_json_bytes,
    validate_marker_changed_paths,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def test_recovery_marker_commit_must_change_exactly_two_identity_files() -> None:
    validate_marker_changed_paths(EXP301R_MARKER_PATHS)
    validate_marker_changed_paths(tuple(reversed(EXP301R_MARKER_PATHS)))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((EXP301R_MARKER_PATHS[0],))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((*EXP301R_MARKER_PATHS, "README.md"))


def test_recovery_identity_from_components_hashes_exact_workflow_bytes() -> None:
    workflow = b"name: recovery\n"
    identity = build_recovery_identity_from_components(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_bytes=workflow,
    )
    assert identity.workflow_sha256 == hashlib.sha256(workflow).hexdigest()
    assert identity.source_commit_sha == _h("a", 40)
    assert len(identity.source_tree_digest) == 64


def test_canonical_marker_json_bytes_are_reproducible_and_newline_terminated() -> None:
    identity = build_recovery_identity_from_components(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_bytes=b"workflow\n",
    )
    first = canonical_marker_json_bytes(identity)
    second = canonical_marker_json_bytes(identity)
    assert first == second
    assert first.endswith(b"\n")
    assert hashlib.sha256(first).hexdigest()
