from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

RECOVERY_SCHEMA = "EXP301R-STANDARD-RUNNER-SHARDED-RECOVERY-V1"
PRIOR_FAILED_RUN_ID = 34823251660
SHARD_COUNT = 16
WORLDS_PER_ROOT = 512
WORLDS_PER_SHARD = WORLDS_PER_ROOT // SHARD_COUNT

_ALLOWED_EXACT_PATHS = frozenset(
    {
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md",
        "docs/superpowers/plans/2026-09-14-exp301r-standard-runner-sharded-recovery.md",
        "scripts/verify_exp301r_recovery.py",
    }
)
_ALLOWED_PREFIXES = (
    "src/nolane_ai/experiments/exp301r_",
    "scripts/exp301r_",
    "tests/test_exp301r_",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def write_once_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_bytes(payload).decode("utf-8"))
        handle.write("\n")


def recovery_beacon(run_id: int | str) -> str:
    text = str(run_id)
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("run_id must be a positive decimal integer")
    return f"github-run-{text}-exp301r-v1"


def challenge_shard_bounds(shard_index: int) -> tuple[int, int]:
    if not isinstance(shard_index, int) or not 0 <= shard_index < SHARD_COUNT:
        raise ValueError(f"shard_index must be in [0,{SHARD_COUNT - 1}]")
    start = shard_index * WORLDS_PER_SHARD
    return start, start + WORLDS_PER_SHARD


def ensure_exact_shard_cover(shard_indices: Iterable[int]) -> tuple[int, ...]:
    materialized = tuple(shard_indices)
    expected = tuple(range(SHARD_COUNT))
    if tuple(sorted(materialized)) != expected or len(materialized) != SHARD_COUNT:
        raise ValueError(f"recovery requires exactly shard indices {expected}")
    return expected


def _path_allowed(path: str) -> bool:
    if path in _ALLOWED_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def validate_recovery_changed_paths(paths: Iterable[str]) -> tuple[str, ...]:
    materialized = tuple(sorted(set(paths)))
    for path in materialized:
        if not _path_allowed(path):
            raise ValueError(f"forbidden recovery path: {path}")
    return materialized
