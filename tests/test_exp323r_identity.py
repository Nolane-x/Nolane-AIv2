from __future__ import annotations

from dataclasses import replace

import pytest

from nolane_ai.experiments.exp323r_identity import (
    build_execution_identity,
    canonical_identity_json_bytes,
    execution_digest,
)


def _identity():
    return build_execution_identity(
        source_commit_sha="a"*40,
        git_tree_sha="b"*40,
        workflow_sha256="c"*64,
    )


def test_repaired_identity_binds_selected_artifact_and_failed_run() -> None:
    i=_identity()
    assert i.failed_exp323_run_id == 35343381108
    assert i.selected_artifact_id == 10547681681
    assert i.selected_checkpoint_sha256 == "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5"
    assert i.selection_lock_digest == "f74a22e4b3202f5157f0ca71faceedf69cbbf172f04a88aeaf8206f03066d011"
    assert canonical_identity_json_bytes(i).endswith(b"\n")


def test_identity_tamper_is_rejected() -> None:
    i=_identity()
    forged=replace(i, selected_artifact_id=1)
    provisional=replace(forged, exp323r_execution_digest="")
    forged=replace(provisional, exp323r_execution_digest=execution_digest(provisional))
    with pytest.raises(ValueError):
        canonical_identity_json_bytes(forged)
