from __future__ import annotations

import hashlib
import pytest

from nolane_ai.experiments.exp324_freeze import (
    BASE_SHA,FROZEN_PATHS,MARKER_PATHS,marker_json_bytes,marker_sidecar_bytes,
    validate_marker_paths,validate_source_paths,
)
from nolane_ai.experiments.exp324_identity import build_execution_identity


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def test_freeze_descends_from_exact_exp323r_marker() -> None:
    assert BASE_SHA=="217477a911aabbb764741e76cca38e54013e7da6"


def test_frozen_allowlist_contains_complete_exp324_surface() -> None:
    required={
        ".github/workflows/exp324-family-isolated-learnability.yml",
        ".github/workflows/v017-exp324-contract.yml",
        ".github/workflows/v017-exp324-runtime.yml",
        "protocols/v017/exp324_preregistration_v1.json",
        "src/nolane_ai/experiments/exp324_contract.py",
        "src/nolane_ai/experiments/exp324_identity.py",
        "src/nolane_ai/experiments/exp324_runtime.py",
        "src/nolane_ai/experiments/exp324_freeze.py",
        "scripts/verify_exp324_freeze.py",
        *MARKER_PATHS,
    }
    assert required<=set(FROZEN_PATHS)
    validate_source_paths(required)
    with pytest.raises(ValueError): validate_source_paths((*required,"README.md"))


def test_marker_is_exactly_two_identity_files() -> None:
    validate_marker_paths(MARKER_PATHS)
    with pytest.raises(ValueError): validate_marker_paths((MARKER_PATHS[0],))


def test_marker_sidecar_is_reproducible() -> None:
    i=identity(); payload=marker_json_bytes(i)
    digest=hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(i)==f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")
