from __future__ import annotations
import hashlib,pytest
from nolane_ai.experiments.exp327_freeze import *
from nolane_ai.experiments.exp327_identity import build_execution_identity
def test_exact_exp326_parent():assert BASE_SHA=="fb66defc39bd9861b9bbaf0e6641cda6ae607b1d"
def test_allowlist_and_marker_topology():
 validate_source_paths(FROZEN_PATHS)
 with pytest.raises(ValueError):validate_source_paths((*FROZEN_PATHS,"README.md"))
 validate_marker_paths(MARKER_PATHS)
 with pytest.raises(ValueError):validate_marker_paths((MARKER_PATHS[0],))
def test_sidecar_reproducible():
 i=build_execution_identity(source_commit_sha="a"*40,git_tree_sha="b"*40,workflow_sha256="c"*64);p=marker_json_bytes(i)
 assert marker_sidecar_bytes(i)==f"{hashlib.sha256(p).hexdigest()}  {MARKER_PATHS[0]}\n".encode()
