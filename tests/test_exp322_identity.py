from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp322_identity import (
    build_exp322_execution_identity,
    canonical_exp322_execution_digest,
    canonical_exp322_identity_json_bytes,
    source_tree_digest_from_git_tree_sha,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def _identity():
    return build_exp322_execution_identity(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        workflow_sha256=_h("c", 64),
    )


def _reseal(identity):
    provisional = replace(identity, exp322_execution_digest="")
    return replace(
        provisional,
        exp322_execution_digest=canonical_exp322_execution_digest(provisional),
    )


def test_identity_binds_source_parent_geometry_and_budget() -> None:
    identity = _identity()
    assert identity.arms == ("HOLD_1E4", "DECAY_5E5")
    assert identity.checkpoints == (1280, 1536, 2048)
    assert identity.device == "cpu"
    assert identity.max_additional_optimizer_steps_per_arm == 1024
    assert len(identity.exp322_execution_digest) == 64
    assert canonical_exp322_identity_json_bytes(identity).endswith(b"\n")


def test_tree_digest_is_domain_separated() -> None:
    first = source_tree_digest_from_git_tree_sha(_h("a", 40))
    assert first != source_tree_digest_from_git_tree_sha(_h("b", 40))
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("bad")


def test_resealed_parent_or_checkpoint_substitution_is_rejected() -> None:
    forged = _reseal(replace(_identity(), parent_exp321_evidence_digest="0" * 64))
    with pytest.raises(ValueError, match="parent EXP-321"):
        canonical_exp322_identity_json_bytes(forged)

    forged = _reseal(replace(_identity(), selected_checkpoint_artifact_id=1))
    with pytest.raises(ValueError, match="checkpoint artifact"):
        canonical_exp322_identity_json_bytes(forged)


def test_resealed_geometry_drift_is_rejected() -> None:
    for forged in (
        _reseal(replace(_identity(), arms=("HOLD_1E4",))),
        _reseal(replace(_identity(), checkpoints=(1280, 2048))),
        _reseal(replace(_identity(), device="cuda")),
        _reseal(replace(_identity(), max_additional_optimizer_steps_per_arm=2048)),
    ):
        with pytest.raises(ValueError):
            canonical_exp322_identity_json_bytes(forged)
