from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp323_identity import (
    build_exp323_execution_identity,
    canonical_exp323_execution_digest,
    canonical_exp323_identity_json_bytes,
    source_tree_digest_from_git_tree_sha,
)


def _identity():
    return build_exp323_execution_identity(
        source_commit_sha="a" * 40,
        git_tree_sha="b" * 40,
        workflow_sha256="c" * 64,
    )


def _reseal(identity):
    provisional = replace(identity, exp323_execution_digest="")
    return replace(
        provisional,
        exp323_execution_digest=canonical_exp323_execution_digest(provisional),
    )


def test_identity_binds_parent_replay_geometry_and_budget() -> None:
    identity = _identity()
    assert identity.arms == ("HOLD_5E5", "DECAY_2P5E5")
    assert identity.checkpoints == (2304, 2560, 3072)
    assert identity.device == "cpu"
    assert identity.max_additional_optimizer_steps_per_arm == 1024
    assert len(identity.exp323_execution_digest) == 64
    assert canonical_exp323_identity_json_bytes(identity).endswith(b"\n")


def test_tree_digest_is_domain_separated() -> None:
    first = source_tree_digest_from_git_tree_sha("a" * 40)
    assert first != source_tree_digest_from_git_tree_sha("b" * 40)
    with pytest.raises(ValueError, match="git tree"):
        source_tree_digest_from_git_tree_sha("bad")


def test_resealed_parent_or_replay_substitution_is_rejected() -> None:
    for forged in (
        _reseal(replace(_identity(), parent_exp322_evidence_digest="0" * 64)),
        _reseal(replace(_identity(), parent_decay_model_state_digest="0" * 64)),
        _reseal(replace(_identity(), source_checkpoint_artifact_id=1)),
        _reseal(replace(_identity(), arms=("HOLD_5E5",))),
        _reseal(replace(_identity(), checkpoints=(2304, 3072))),
        _reseal(replace(_identity(), device="cuda")),
    ):
        with pytest.raises(ValueError):
            canonical_exp323_identity_json_bytes(forged)
