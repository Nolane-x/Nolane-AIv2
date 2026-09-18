from __future__ import annotations

import hashlib
import pytest

from nolane_ai.experiments.exp323r_freeze import (
    BASE_SHA,FROZEN_PATHS,MARKER_PATHS,marker_json_bytes,marker_sidecar_bytes,
    validate_marker_paths,validate_source_paths,
)
from nolane_ai.experiments.exp323r_identity import build_execution_identity


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def test_repair_freeze_descends_from_exact_exp323_marker() -> None:
    assert BASE_SHA=="45b539b5c69c982601ced9d642935472cc8458e1"


def test_repair_freeze_allowlist_contains_complete_surface() -> None:
    required={
        ".github/workflows/exp323r-second-decay-convergence.yml",
        ".github/workflows/exp323r-reconstruction-materialization.yml",
        "protocols/v017/exp323r_reconstruction_selection_lock_v1.json",
        "src/nolane_ai/experiments/exp323r_repair.py",
        "src/nolane_ai/experiments/exp323r_identity.py",
        "src/nolane_ai/experiments/exp323r_freeze.py",
        "scripts/verify_exp323r_freeze.py",
        *MARKER_PATHS,
    }
    assert required<=set(FROZEN_PATHS)
    validate_source_paths(required)
    with pytest.raises(ValueError): validate_source_paths((*required,"README.md"))


def test_repair_marker_is_exactly_two_identity_files() -> None:
    validate_marker_paths(MARKER_PATHS)
    with pytest.raises(ValueError): validate_marker_paths((MARKER_PATHS[0],))


def test_marker_sidecar_is_reproducible() -> None:
    i=identity(); payload=marker_json_bytes(i)
    digest=hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(i)==f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")
