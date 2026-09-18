from __future__ import annotations

from dataclasses import replace
import pytest

from nolane_ai.experiments.exp324_identity import (
    build_execution_identity,canonical_identity_json_bytes,execution_digest,
)


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def reseal(i):
    p=replace(i,exp324_execution_digest="")
    return replace(p,exp324_execution_digest=execution_digest(p))


def test_identity_binds_parent_recovery_and_matched_family_geometry() -> None:
    i=identity()
    assert i.parent_recovery_run_id==35356470634
    assert i.parent_final_artifact_id==10552276255
    assert i.reconstruction_artifact_id==10547681681
    assert i.updates_per_family==256
    assert i.local_checkpoints==(64,128,256)
    assert canonical_identity_json_bytes(i).endswith(b"\n")


def test_resealed_parent_or_budget_drift_is_rejected() -> None:
    for forged in (
        reseal(replace(identity(),parent_final_artifact_id=1)),
        reseal(replace(identity(),updates_per_family=1024)),
        reseal(replace(identity(),families=("bad",))),
    ):
        with pytest.raises(ValueError): canonical_identity_json_bytes(forged)
