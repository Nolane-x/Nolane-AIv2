from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp301_freeze import (
    EXP301_MARKER_PATHS,
    build_freeze_identity_from_git,
    canonical_identity_json_bytes,
    validate_marker_changed_paths,
)


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


def verify_freeze(repo_root: Path) -> None:
    identity_path = repo_root / EXP301_MARKER_PATHS[0]
    sidecar_path = repo_root / EXP301_MARKER_PATHS[1]
    if not identity_path.is_file() or not sidecar_path.is_file():
        raise ValueError("EXP-301 freeze marker files are missing")

    payload = identity_path.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("EXP-301 freeze identity must be a JSON object")
    source_commit = raw.get("source_commit_sha")
    if not isinstance(source_commit, str):
        raise ValueError("EXP-301 freeze identity is missing source_commit_sha")

    expected_sidecar = f"{hashlib.sha256(payload).hexdigest()}  {EXP301_MARKER_PATHS[0]}\n"
    if sidecar_path.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError("EXP-301 freeze sidecar digest mismatch")

    current_head = _git(repo_root, "rev-parse", "HEAD")
    parent = _git(repo_root, "rev-parse", "HEAD^")
    if parent != source_commit:
        raise ValueError(
            "EXP-301 marker commit parent must equal the frozen pre-marker source commit"
        )
    changed_paths = tuple(
        item for item in _git(repo_root, "diff", "--name-only", source_commit, current_head).splitlines() if item
    )
    validate_marker_changed_paths(changed_paths)

    rebuilt = build_freeze_identity_from_git(repo_root, source_commit)
    expected_payload = canonical_identity_json_bytes(rebuilt)
    if payload != expected_payload:
        raise ValueError("EXP-301 freeze identity does not reproduce from frozen Git objects")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the EXP-301 marker-only pre-data freeze")
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_freeze(Path(args.repo_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
