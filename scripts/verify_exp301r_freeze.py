from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

from nolane_ai.experiments.exp301r_freeze import (
    EXP301R_MARKER_PATHS,
    build_recovery_identity_from_git,
    canonical_marker_json_bytes,
    validate_marker_changed_paths,
)
from nolane_ai.experiments.exp301r_identity import (
    ORIGINAL_EXP301_MARKER_SHA,
    RecoveryExecutionIdentity,
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


def verify_recovery_freeze(repo_root: Path) -> None:
    identity_path = repo_root / EXP301R_MARKER_PATHS[0]
    sidecar_path = repo_root / EXP301R_MARKER_PATHS[1]
    if not identity_path.is_file() or not sidecar_path.is_file():
        raise ValueError("EXP-301R recovery marker files are missing")

    payload = identity_path.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("EXP-301R recovery identity must be a JSON object")
    expected_fields = {field.name for field in fields(RecoveryExecutionIdentity)}
    if set(raw) != expected_fields:
        raise ValueError("EXP-301R recovery identity field set mismatch")
    try:
        identity = RecoveryExecutionIdentity(**raw)
    except TypeError as exc:
        raise ValueError("EXP-301R recovery identity schema mismatch") from exc
    if canonical_marker_json_bytes(identity) != payload:
        raise ValueError("EXP-301R recovery identity canonical payload mismatch")

    expected_sidecar = f"{hashlib.sha256(payload).hexdigest()}  {EXP301R_MARKER_PATHS[0]}\n"
    if sidecar_path.read_text(encoding="ascii") != expected_sidecar:
        raise ValueError("EXP-301R recovery sidecar digest mismatch")

    source_commit = identity.source_commit_sha
    current_head = _git(repo_root, "rev-parse", "HEAD")
    parent = _git(repo_root, "rev-parse", "HEAD^")
    if parent != source_commit:
        raise ValueError("EXP-301R marker commit parent must equal recovery source commit")

    subprocess.run(
        ("git", "merge-base", "--is-ancestor", ORIGINAL_EXP301_MARKER_SHA, source_commit),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    changed_paths = tuple(
        line
        for line in _git(repo_root, "diff", "--name-only", source_commit, current_head).splitlines()
        if line
    )
    validate_marker_changed_paths(changed_paths)

    rebuilt = build_recovery_identity_from_git(repo_root, source_commit)
    if canonical_marker_json_bytes(rebuilt) != payload:
        raise ValueError("EXP-301R recovery identity does not reproduce from frozen Git objects")


def main() -> int:
    verify_recovery_freeze(Path(".").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
