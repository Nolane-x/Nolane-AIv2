from dataclasses import asdict

import pytest

from nolane_ai.experiments.exp301r_identity import (
    EXP301R_IDENTITY_SCHEMA,
    ORIGINAL_EXP301_MARKER_SHA,
    build_recovery_execution_identity,
    canonical_recovery_identity_json_bytes,
    source_tree_digest_from_git_tree_sha,
)
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_IMPLEMENTATION_DIGEST,
    PRIOR_FAILED_RUN_ID,
    RECOVERY_SCHEMA,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def test_source_tree_digest_binds_git_tree_sha() -> None:
    digest = source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert len(digest) == 64
    assert digest == source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert digest != source_tree_digest_from_git_tree_sha(_h("b", 40))
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("bad")


def test_recovery_execution_identity_binds_frozen_science_and_execution_geometry() -> None:
    identity = build_recovery_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )
    assert identity.schema == EXP301R_IDENTITY_SCHEMA
    assert identity.recovery_schema == RECOVERY_SCHEMA
    assert identity.original_exp301_marker_sha == ORIGINAL_EXP301_MARKER_SHA
    assert identity.frozen_implementation_digest == FROZEN_IMPLEMENTATION_DIGEST
    assert identity.prior_failed_run_id == PRIOR_FAILED_RUN_ID
    assert identity.shard_count == 16
    assert identity.worlds_per_shard == 32
    assert identity.device == "cpu"
    assert len(identity.recovery_execution_digest) == 64


def test_recovery_execution_digest_changes_if_source_or_workflow_changes() -> None:
    a = build_recovery_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )
    b = build_recovery_execution_identity(
        source_commit_sha=_h("d", 40),
        git_tree_sha=_h("e", 40),
        workflow_sha256=_h("c", 64),
    )
    c = build_recovery_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("f", 64),
    )
    assert a.recovery_execution_digest != b.recovery_execution_digest
    assert a.recovery_execution_digest != c.recovery_execution_digest


def test_recovery_identity_canonical_json_is_compact_sorted_and_newline_terminated() -> None:
    identity = build_recovery_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )
    payload = canonical_recovery_identity_json_bytes(identity)
    assert payload.endswith(b"\n")
    assert b"  " not in payload
    assert payload == canonical_recovery_identity_json_bytes(identity)
    assert b'"recovery_execution_digest"' in payload
    assert set(asdict(identity)) == {
        "schema",
        "recovery_schema",
        "original_exp301_marker_sha",
        "frozen_implementation_digest",
        "prior_failed_run_id",
        "source_commit_sha",
        "source_tree_digest",
        "workflow_sha256",
        "shard_count",
        "worlds_per_shard",
        "device",
        "recovery_execution_digest",
    }


def test_recovery_identity_rejects_malformed_git_and_workflow_hashes() -> None:
    with pytest.raises(ValueError, match="source commit"):
        build_recovery_execution_identity(
            source_commit_sha="bad",
            git_tree_sha=_h("b", 40),
            workflow_sha256=_h("c", 64),
        )
    with pytest.raises(ValueError, match="workflow"):
        build_recovery_execution_identity(
            source_commit_sha=_h("a", 40),
            git_tree_sha=_h("b", 40),
            workflow_sha256="bad",
        )
