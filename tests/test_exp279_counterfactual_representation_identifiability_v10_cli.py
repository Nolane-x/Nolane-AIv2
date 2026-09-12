from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _help(script: str) -> str:
    path = SCRIPTS / script
    assert path.exists(), f"missing V10 CLI: {path}"
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_v10_shard_cli_exposes_only_frozen_identity_inputs() -> None:
    text = _help("run_exp279_counterfactual_representation_identifiability_v10_dev.py")
    for flag in (
        "--train-replicates",
        "--canonical-index",
        "--scientific-branch-head",
        "--executed-commit",
        "--output",
    ):
        assert flag in text
    for forbidden in (
        "--fit-replicates",
        "--heldout-replicates",
        "--root-prefix",
        "--query-chunk-size",
        "--k",
        "--threshold",
        "--representation",
    ):
        assert forbidden not in text


def test_v10_budget_cli_accepts_only_four_shards_and_output() -> None:
    text = _help("aggregate_exp279_counterfactual_representation_identifiability_v10_dev.py")
    assert "--shard" in text
    assert "--output" in text
    assert "--train-replicates" not in text
    assert "--canonical-index" not in text


def test_v10_cross_cli_accepts_only_two_budget_receipts_and_output() -> None:
    text = _help("classify_exp279_counterfactual_representation_identifiability_v10_dev.py")
    for flag in ("--train60", "--train120", "--output"):
        assert flag in text
    for forbidden in ("--threshold", "--representation", "--allow-partial"):
        assert forbidden not in text
