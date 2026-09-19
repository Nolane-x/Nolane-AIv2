from __future__ import annotations
import hashlib,subprocess
from pathlib import Path
from typing import Iterable
from .exp327_identity import build_execution_identity,canonical_identity_json_bytes

BASE_SHA="fb66defc39bd9861b9bbaf0e6641cda6ae607b1d"
WORKFLOW_PATH=".github/workflows/exp327-iterative-minimal-subset.yml"
MARKER_PATHS=("protocols/v017/exp327_execution_identity_v1.json","protocols/v017/exp327_execution_identity_v1.sha256")
FROZEN_PATHS=(
 ".github/workflows/exp327-iterative-minimal-subset.yml",".github/workflows/v017-exp327-contract.yml",
 "docs/superpowers/specs/2026-09-19-exp327-iterative-minimal-failing-subset.md",
 "protocols/v017/exp327_preregistration_v1.json","protocols/v017/exp327_preregistration_v1.sha256",
 "scripts/exp327_run.py","scripts/verify_exp327_freeze.py",
 "src/nolane_ai/experiments/exp327_contract.py","src/nolane_ai/experiments/exp327_identity.py","src/nolane_ai/experiments/exp327_runtime.py","src/nolane_ai/experiments/exp327_freeze.py",
 "tests/test_exp327_contract.py","tests/test_exp327_identity.py","tests/test_exp327_runtime.py","tests/test_exp327_workflow.py","tests/test_exp327_freeze.py",*MARKER_PATHS)
def _git(root,*args):return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
def _gb(root,*args):return subprocess.check_output(["git","-C",str(root),*args])
def build_identity_from_git(repo_root,source_commit_sha="HEAD"):
 root=Path(repo_root).resolve();commit=_git(root,"rev-parse",source_commit_sha);tree=_git(root,"rev-parse",f"{commit}^{{tree}}");wf=_gb(root,"show",f"{commit}:{WORKFLOW_PATH}")
 return build_execution_identity(source_commit_sha=commit,git_tree_sha=tree,workflow_sha256=hashlib.sha256(wf).hexdigest())
def marker_json_bytes(i):return canonical_identity_json_bytes(i)
def marker_sidecar_bytes(i):
 p=marker_json_bytes(i);return f"{hashlib.sha256(p).hexdigest()}  {MARKER_PATHS[0]}\n".encode()
def validate_source_paths(paths:Iterable[str]):
 bad=sorted(set(paths)-set(FROZEN_PATHS))
 if bad:raise ValueError(f"unrelated EXP-327 changed paths: {bad}")
def validate_marker_paths(paths:Iterable[str]):
 x=tuple(paths)
 if len(x)!=2 or set(x)!=set(MARKER_PATHS):raise ValueError("EXP-327 marker-only commit must change exactly two identity files")
