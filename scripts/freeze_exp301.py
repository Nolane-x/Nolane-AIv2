from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp301_freeze import (
    EXP301_MARKER_PATHS,
    build_freeze_identity_from_components,
    canonical_identity_json_bytes,
)


WORLD_PATH = "src/nolane_ai/experiments/exp301_worlds.py"
ANALYSIS_PATHS = (
    "src/nolane_ai/experiments/exp301_analysis.py",
    "src/nolane_ai/experiments/exp301_cross_root.py",
    "src/nolane_ai/experiments/exp301_evidence.py",
    "src/nolane_ai/experiments/exp301_evaluation.py",
    "src/nolane_ai/experiments/exp301_compute.py",
    "src/nolane_ai/experiments/exp301_execution.py",
    "src/nolane_ai/experiments/exp301_identity.py",
)
WORKFLOW_PATH = ".github/workflows/exp301-scientific-court.yml"


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _blob_sha(repo_root: Path, commit_sha: str, path: str) -> str:
    return _git(repo_root, "rev-parse", f"{commit_sha}:{path}")


def build_identity_from_git(repo_root: Path, source_commit_sha: str):
    source_commit_sha = _git(repo_root, "rev-parse", source_commit_sha)
    git_tree_sha = _git(repo_root, "rev-parse", f"{source_commit_sha}^{{tree}}")
    return build_freeze_identity_from_components(
        source_commit_sha=source_commit_sha,
        git_tree_sha=git_tree_sha,
        worlds_blob_sha=_blob_sha(repo_root, source_commit_sha, WORLD_PATH),
        analysis_blob_shas=tuple(
            _blob_sha(repo_root, source_commit_sha, path) for path in ANALYSIS_PATHS
        ),
        workflow_blob_sha=_blob_sha(repo_root, source_commit_sha, WORKFLOW_PATH),
    )


def write_freeze_files(repo_root: Path, source_commit_sha: str) -> None:
    identity = build_identity_from_git(repo_root, source_commit_sha)
    identity_path = repo_root / EXP301_MARKER_PATHS[0]
    sidecar_path = repo_root / EXP301_MARKER_PATHS[1]
    if identity_path.exists() or sidecar_path.exists():
        raise FileExistsError("EXP-301 freeze marker files are write-once")
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_identity_json_bytes(identity)
    identity_path.write_bytes(payload)
    raw_digest = hashlib.sha256(payload).hexdigest()
    sidecar_path.write_text(
        f"{raw_digest}  {EXP301_MARKER_PATHS[0]}\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the EXP-301 pre-data freeze marker files")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-commit", default="HEAD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    write_freeze_files(repo_root, args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
