from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SHARD_SCRIPT = ROOT / "scripts" / "run_exp279_powered_support_stability_v8_dev.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_exp279_powered_support_stability_v8_dev.py"
BRANCH_HEAD = "a" * 40
EXECUTED_COMMIT = "b" * 40


def _assert_sidecar(output: Path) -> None:
    raw = output.read_bytes()
    sidecar = output.with_name(output.name + ".sha256")
    assert sidecar.read_text(encoding="utf-8") == hashlib.sha256(raw).hexdigest() + "\n"


def test_v8_shard_cli_writes_byte_sha256_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "shard.json"
    subprocess.run(
        [
            sys.executable,
            str(SHARD_SCRIPT),
            "--tiny-contract",
            "--train-replicates", "60",
            "--canonical-index", "0",
            "--scientific-branch-head", BRANCH_HEAD,
            "--executed-commit", EXECUTED_COMMIT,
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    _assert_sidecar(output)


def test_v8_budget_cli_writes_byte_sha256_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("v8_budget_cli", AGGREGATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    shard_paths = []
    for index in range(4):
        path = tmp_path / f"shard-{index}.json"
        path.write_text(json.dumps({"canonical_index": index}) + "\n", encoding="utf-8")
        shard_paths.append(path)

    receipt = {
        "artifact_digest": "d" * 64,
        "canonical_prevalence_floor": 0.01,
        "train_cell_classification": "SUPPORT_RECURRENT",
        "train_replicates": 60,
    }
    monkeypatch.setattr(module, "aggregate_exp279_powered_support_stability_v8_budget", lambda shards: receipt)
    output = tmp_path / "budget.json"
    argv = [str(AGGREGATE_SCRIPT)]
    for path in shard_paths:
        argv.extend(["--shard", str(path)])
    argv.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0
    _assert_sidecar(output)
