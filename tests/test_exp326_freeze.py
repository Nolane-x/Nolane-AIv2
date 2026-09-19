from __future__ import annotations

import hashlib
import pytest

from nolane_ai.experiments.exp326_freeze import (
    BASE_SHA,FROZEN_PATHS,MARKER_PATHS,marker_json_bytes,marker_sidecar_bytes,
    validate_marker_paths,validate_source_paths,
)
from nolane_ai.experiments.exp326_identity import build_execution_identity


def identity():
    return build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64)


def test_freeze_descends_from_exact_exp325_marker() -> None:
    assert BASE_SHA=="b24830c842258258f42ffb92e5f00277f8db8e9b"


def test_freeze_allowlist_contains_complete_exp326_surface() -> None:
    required={
        ".github/workflows/exp326-balanced-multiworld-breakpoint.yml",
        ".github/workflows/v017-exp326-contract.yml",
        "protocols/v017/exp326_preregistration_v1.json",
        "src/nolane_ai/experiments/exp326_contract.py",
        "src/nolane_ai/experiments/exp326_identity.py",
        "src/nolane_ai/experiments/exp326_runtime.py",
        "src/nolane_ai/experiments/exp326_freeze.py",
        "scripts/verify_exp326_freeze.py",
        *MARKER_PATHS,
    }
    assert required<=set(FROZEN_PATHS)
    validate_source_paths(required)
    with pytest.raises(ValueError):
        validate_source_paths((*required,"README.md"))


def test_marker_is_exactly_two_identity_files() -> None:
    validate_marker_paths(MARKER_PATHS)
    with pytest.raises(ValueError):
        validate_marker_paths((MARKER_PATHS[0],))


def test_marker_sidecar_is_reproducible() -> None:
    i=identity(); payload=marker_json_bytes(i)
    digest=hashlib.sha256(payload).hexdigest()
    assert marker_sidecar_bytes(i)==f"{digest}  {MARKER_PATHS[0]}\n".encode("ascii")
