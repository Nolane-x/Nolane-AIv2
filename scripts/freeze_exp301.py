from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from nolane_ai.experiments.exp301_freeze import (
    EXP301_MARKER_PATHS,
    build_freeze_identity_from_git,
    canonical_identity_json_bytes,
)


def write_freeze_files(repo_root: Path, source_commit_sha: str) -> None:
    identity = build_freeze_identity_from_git(repo_root, source_commit_sha)
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
    write_freeze_files(Path(args.repo_root).resolve(), args.source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
