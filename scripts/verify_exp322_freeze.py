from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp322_freeze import (
    EXP322_BASE_SHA,
    EXP322_WORKFLOW_PATH,
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_frozen_changed_paths,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp322_identity import (
    Exp322ExecutionIdentity,
    source_tree_digest_from_git_tree_sha,
    validate_exp322_execution_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify sealed EXP-322 execution identity")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--marker-json", required=True)
    parser.add_argument("--marker-sha256", required=True)
    return parser


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo_root).resolve()
    payload = json.loads(Path(args.marker_json).read_text(encoding="utf-8"))
    expected_fields = {item.name for item in fields(Exp322ExecutionIdentity)}
    if set(payload) != expected_fields:
        raise SystemExit("EXP-322 marker identity field mismatch")
    identity = Exp322ExecutionIdentity(**payload)
    try:
        validate_exp322_execution_identity(identity)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if identity.source_commit_sha != args.source_commit_sha:
        raise SystemExit("EXP-322 marker source commit mismatch")
    if Path(args.marker_json).read_bytes() != canonical_marker_json_bytes(identity):
        raise SystemExit("EXP-322 marker JSON is not canonical")
    if Path(args.marker_sha256).read_bytes() != marker_sidecar_bytes(identity):
        raise SystemExit("EXP-322 marker SHA sidecar mismatch")

    tree_sha = _git(repo, "rev-parse", f"{args.source_commit_sha}^{{tree}}")
    if identity.source_tree_digest != source_tree_digest_from_git_tree_sha(tree_sha):
        raise SystemExit("EXP-322 source tree digest mismatch")
    workflow = _git_bytes(repo, "show", f"{args.source_commit_sha}:{EXP322_WORKFLOW_PATH}")
    if hashlib.sha256(workflow).hexdigest() != identity.workflow_sha256:
        raise SystemExit("EXP-322 sealed workflow digest mismatch")

    merge_base = _git(repo, "merge-base", EXP322_BASE_SHA, args.source_commit_sha)
    if merge_base != EXP322_BASE_SHA:
        raise SystemExit("EXP-322 source is not descended from sealed EXP-321 marker")
    changed = tuple(
        line
        for line in _git(repo, "diff", "--name-only", EXP322_BASE_SHA, args.source_commit_sha).splitlines()
        if line
    )
    validate_frozen_changed_paths(changed)

    parent = _git(repo, "rev-parse", "HEAD^")
    if parent != args.source_commit_sha:
        raise SystemExit("EXP-322 marker parent is not the sealed source commit")
    marker_changed = tuple(
        line
        for line in _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
        if line
    )
    validate_marker_changed_paths(marker_changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
