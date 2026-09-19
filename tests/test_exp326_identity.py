from __future__ import annotations

from dataclasses import replace
import pytest

from nolane_ai.experiments.exp326_identity import (
    build_execution_identity, canonical_identity_json_bytes, execution_digest,
)


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def test_identity_binds_parent_groups_and_exposure_geometry() -> None:
    i=identity()
    assert i.parent_run_id==35406123945
    assert i.parent_final_artifact_id==10572536792
    assert i.group_ids==("P01","P23","P45","P67","Q0123","Q4567","O01234567")
    assert i.exposure_checkpoints==(8,16,32)
    assert i.exposures_per_world==32
    assert canonical_identity_json_bytes(i).endswith(b"\n")


def test_resealed_parent_substitution_is_rejected() -> None:
    i=replace(identity(),parent_final_artifact_id=1)
    provisional=replace(i,exp326_execution_digest="")
    forged=replace(provisional,exp326_execution_digest=execution_digest(provisional))
    with pytest.raises(ValueError): canonical_identity_json_bytes(forged)
