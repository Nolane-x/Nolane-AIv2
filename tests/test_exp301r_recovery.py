from __future__ import annotations

import json
from pathlib import Path

import pytest

from nolane_ai.experiments.exp301r_recovery import (
    PRIOR_FAILED_RUN_ID,
    RECOVERY_SCHEMA,
    SHARD_COUNT,
    WORLDS_PER_ROOT,
    WORLDS_PER_SHARD,
    canonical_digest,
    challenge_shard_bounds,
    ensure_exact_shard_cover,
    recovery_beacon,
    validate_recovery_changed_paths,
    write_once_json,
)


def test_recovery_constants_bind_failed_attempt_and_fixed_geometry() -> None:
    assert RECOVERY_SCHEMA == "EXP301R-STANDARD-RUNNER-SHARDED-RECOVERY-V1"
    assert PRIOR_FAILED_RUN_ID == 34823251660
    assert SHARD_COUNT == 16
    assert WORLDS_PER_ROOT == 512
    assert WORLDS_PER_SHARD == 32


def test_recovery_beacon_is_stable_for_run_id_and_excludes_attempt() -> None:
    assert recovery_beacon("34899900000") == "github-run-34899900000-exp301r-v1"
    assert recovery_beacon(34899900000) == "github-run-34899900000-exp301r-v1"
    with pytest.raises(ValueError, match="run_id"):
        recovery_beacon("")
    with pytest.raises(ValueError, match="run_id"):
        recovery_beacon("not-a-run")


def test_challenge_shard_bounds_cover_exactly_512_worlds_without_overlap() -> None:
    assert challenge_shard_bounds(0) == (0, 32)
    assert challenge_shard_bounds(15) == (480, 512)
    assert tuple(challenge_shard_bounds(i) for i in range(16)) == tuple(
        (i * 32, (i + 1) * 32) for i in range(16)
    )
    assert ensure_exact_shard_cover(tuple(range(16))) == tuple(range(16))

    with pytest.raises(ValueError, match="shard_index"):
        challenge_shard_bounds(-1)
    with pytest.raises(ValueError, match="shard_index"):
        challenge_shard_bounds(16)
    with pytest.raises(ValueError, match="exactly"):
        ensure_exact_shard_cover(tuple(range(15)))
    with pytest.raises(ValueError, match="exactly"):
        ensure_exact_shard_cover(tuple(range(16)) + (15,))


def test_canonical_digest_is_order_independent_for_object_keys() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert len(canonical_digest({"a": 1})) == 64


def test_write_once_json_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = {"b": 2, "a": 1}
    write_once_json(target, payload)
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        write_once_json(target, payload)


def test_recovery_changed_path_allowlist_rejects_scientific_core_modification() -> None:
    allowed = (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md",
        "docs/superpowers/plans/2026-09-14-exp301r-standard-runner-sharded-recovery.md",
        "src/nolane_ai/experiments/exp301r_recovery.py",
        "scripts/exp301r_train_trial.py",
        "scripts/verify_exp301r_recovery.py",
        "tests/test_exp301r_recovery.py",
    )
    assert validate_recovery_changed_paths(allowed) == tuple(sorted(allowed))

    forbidden = allowed + ("src/nolane_ai/experiments/exp301_scientific.py",)
    with pytest.raises(ValueError, match="forbidden recovery path"):
        validate_recovery_changed_paths(forbidden)


def test_recovery_changed_path_allowlist_rejects_unrelated_files() -> None:
    with pytest.raises(ValueError, match="forbidden recovery path"):
        validate_recovery_changed_paths(("README.md",))
