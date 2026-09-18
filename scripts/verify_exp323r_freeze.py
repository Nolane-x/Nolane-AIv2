from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp323r_freeze import (
    BASE_SHA, WORKFLOW_PATH, marker_json_bytes, marker_sidecar_bytes,
    validate_marker_paths, validate_source_paths,
)
from nolane_ai.experiments.exp323r_identity import (
    Exp323RExecutionIdentity, source_tree_digest_from_git_tree_sha, validate_execution_identity,
)
from nolane_ai.experiments.exp323r_repair import validate_selection_lock


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",required=True)
    p.add_argument("--source-commit-sha",required=True)
    p.add_argument("--marker-json",required=True)
    p.add_argument("--marker-sha256",required=True)
    return p


def git(root: Path,*args:str)->str:
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def git_bytes(root: Path,*args:str)->bytes:
    return subprocess.check_output(["git","-C",str(root),*args])


def main(argv=None)->int:
    a=parser().parse_args(argv); root=Path(a.repo_root).resolve()
    payload=json.loads(Path(a.marker_json).read_text(encoding="utf-8"))
    if set(payload)!={f.name for f in fields(Exp323RExecutionIdentity)}:
        raise SystemExit("EXP-323R marker identity field mismatch")
    identity=Exp323RExecutionIdentity(**payload)
    try: validate_execution_identity(identity)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    if identity.source_commit_sha!=a.source_commit_sha:
        raise SystemExit("EXP-323R marker source commit mismatch")
    if Path(a.marker_json).read_bytes()!=marker_json_bytes(identity):
        raise SystemExit("EXP-323R marker JSON is not canonical")
    if Path(a.marker_sha256).read_bytes()!=marker_sidecar_bytes(identity):
        raise SystemExit("EXP-323R marker sidecar mismatch")
    tree=git(root,"rev-parse",f"{a.source_commit_sha}^{{tree}}")
    if identity.source_tree_digest!=source_tree_digest_from_git_tree_sha(tree):
        raise SystemExit("EXP-323R source tree digest mismatch")
    workflow=git_bytes(root,"show",f"{a.source_commit_sha}:{WORKFLOW_PATH}")
    if hashlib.sha256(workflow).hexdigest()!=identity.workflow_sha256:
        raise SystemExit("EXP-323R workflow digest mismatch")
    if git(root,"merge-base",BASE_SHA,a.source_commit_sha)!=BASE_SHA:
        raise SystemExit("EXP-323R source is not descended from sealed EXP-323 marker")
    changed=tuple(x for x in git(root,"diff","--name-only",BASE_SHA,a.source_commit_sha).splitlines() if x)
    validate_source_paths(changed)
    source_lock=json.loads(git_bytes(root,"show",f"{a.source_commit_sha}:protocols/v017/exp323r_reconstruction_selection_lock_v1.json"))
    validate_selection_lock(source_lock)
    if git(root,"rev-parse","HEAD^")!=a.source_commit_sha:
        raise SystemExit("EXP-323R marker parent mismatch")
    marker_changed=tuple(x for x in git(root,"diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines() if x)
    validate_marker_paths(marker_changed)
    return 0


if __name__=="__main__": raise SystemExit(main())
