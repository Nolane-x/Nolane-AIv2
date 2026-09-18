from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp321_contract import (
    EXP319_FINAL_EVIDENCE_DIGEST,
    SELECTED_CHECKPOINT_ARTIFACT_ID,
    SELECTED_CHECKPOINT_ZIP_DIGEST,
    SELECTED_MODEL_STATE_DIGEST,
    SELECTED_RECEIPT_ARTIFACT_DIGEST,
    preregistration_digest,
)
from nolane_ai.experiments.exp321_identity import (
    EXP321_IDENTITY_SCHEMA,
    build_exp321_execution_identity,
    canonical_exp321_identity_json_bytes,
    source_tree_digest_from_git_tree_sha,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def _identity():
    return build_exp321_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )


def test_identity_binds_parent_checkpoint_preregistration_and_zero_gradient() -> None:
    identity = _identity()
    assert identity.schema == EXP321_IDENTITY_SCHEMA
    assert identity.approved_preregistration_digest == preregistration_digest()
    assert identity.parent_exp319_final_evidence_digest == EXP319_FINAL_EVIDENCE_DIGEST
    assert identity.selected_checkpoint_artifact_id == SELECTED_CHECKPOINT_ARTIFACT_ID
    assert identity.selected_checkpoint_zip_digest == SELECTED_CHECKPOINT_ZIP_DIGEST
    assert identity.selected_receipt_artifact_digest == SELECTED_RECEIPT_ARTIFACT_DIGEST
    assert identity.selected_model_state_digest == SELECTED_MODEL_STATE_DIGEST
    assert identity.effort_grid == (1, 2, 4, 8)
    assert identity.device == "cpu"
    assert identity.gradient_updates == 0
    assert len(identity.exp321_execution_digest) == 64


def test_tree_digest_is_domain_separated() -> None:
    first = source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert first == source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert first != source_tree_digest_from_git_tree_sha(_h("b", 40))
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("bad")


def test_execution_digest_changes_with_source_or_workflow() -> None:
    base = _identity()
    source_changed = build_exp321_execution_identity(
        source_commit_sha=_h("d", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )
    tree_changed = build_exp321_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("d", 40),
        workflow_sha256=_h("c", 64),
    )
    workflow_changed = build_exp321_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("d", 64),
    )
    assert source_changed.exp321_execution_digest != base.exp321_execution_digest
    assert tree_changed.exp321_execution_digest != base.exp321_execution_digest
    assert workflow_changed.exp321_execution_digest != base.exp321_execution_digest


def test_canonical_identity_is_digest_checked() -> None:
    identity = _identity()
    payload = canonical_exp321_identity_json_bytes(identity)
    assert payload.endswith(b"\n")
    assert b"  " not in payload
    with pytest.raises(ValueError, match="execution digest"):
        canonical_exp321_identity_json_bytes(
            replace(identity, exp321_execution_digest="0" * 64)
        )
