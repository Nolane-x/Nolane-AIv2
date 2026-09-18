from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp321_freeze import (
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp321_identity import (
    Exp321ExecutionIdentity,
    source_tree_digest_from_git_tree_sha,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify sealed EXP-321 execution identity")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--marker-json", required=True)
    parser.add_argument("--marker-sha256", required=True)
    return parser


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo_root).resolve()
    payload = json.loads(Path(args.marker_json).read_text(encoding="utf-8"))
    expected_fields = {item.name for item in fields(Exp321ExecutionIdentity)}
    if set(payload) != expected_fields:
        raise SystemExit("EXP-321 marker identity field mismatch")
    identity = Exp321ExecutionIdentity(**payload)
    if identity.source_commit_sha != args.source_commit_sha:
        raise SystemExit("EXP-321 marker source commit mismatch")
    if Path(args.marker_json).read_bytes() != canonical_marker_json_bytes(identity):
        raise SystemExit("EXP-321 marker JSON is not canonical")
    if Path(args.marker_sha256).read_bytes() != marker_sidecar_bytes(identity):
        raise SystemExit("EXP-321 marker SHA sidecar mismatch")
    tree_sha = _git(repo, "rev-parse", f"{args.source_commit_sha}^{{tree}}")
    if identity.source_tree_digest != source_tree_digest_from_git_tree_sha(tree_sha):
        raise SystemExit("EXP-321 source tree digest mismatch")
    parent = _git(repo, "rev-parse", "HEAD^")
    if parent != args.source_commit_sha:
        raise SystemExit("EXP-321 marker parent is not the sealed source commit")
    changed = tuple(
        line
        for line in _git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ).splitlines()
        if line
    )
    validate_marker_changed_paths(changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
