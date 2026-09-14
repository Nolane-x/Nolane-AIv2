from __future__ import annotations

from pathlib import Path
import subprocess

from nolane_ai.experiments.exp301r_freeze import EXP301R_MARKER_PATHS
from nolane_ai.experiments.exp301r_recovery import validate_recovery_changed_paths

EXP301_MARKER_SHA = "bac51c29c46e4c1fb3db5445a299da4674fdb6d8"


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_recovery_isolation(repo_root: Path) -> None:
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", EXP301_MARKER_SHA, "HEAD"),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    changed = tuple(
        line
        for line in _git(repo_root, "diff", "--name-only", EXP301_MARKER_SHA, "HEAD").splitlines()
        if line
    )
    recovery_source_paths = tuple(path for path in changed if path not in EXP301R_MARKER_PATHS)
    validate_recovery_changed_paths(recovery_source_paths)
    marker_paths = tuple(path for path in changed if path in EXP301R_MARKER_PATHS)
    if marker_paths and set(marker_paths) != set(EXP301R_MARKER_PATHS):
        raise ValueError("EXP-301R recovery marker paths must appear together")


def main() -> int:
    verify_recovery_isolation(Path(".").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
