from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_outcome_quartet_v9_receipts import canonical_receipt_bytes

ROOT = Path(__file__).resolve().parents[1]
SHARD_SCRIPT = ROOT / "scripts" / "run_exp279_counterfactual_outcome_quartet_v9_dev.py"
BUDGET_SCRIPT = ROOT / "scripts" / "aggregate_exp279_counterfactual_outcome_quartet_v9_dev.py"
CROSS_SCRIPT = ROOT / "scripts" / "classify_exp279_counterfactual_outcome_quartet_v9_dev.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v9_shard_cli_exposes_only_frozen_scientific_identity_knobs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    help_text = completed.stdout.lower()
    for required in (
        "--train-replicates",
        "--canonical-index",
        "--scientific-branch-head",
        "--executed-commit",
        "--output",
    ):
        assert required in help_text
    for forbidden in (
        "--eval-",
        "confirmatory",
        "challenge",
        "fresh-lineage",
        "fresh-evaluation",
        "60000",
        "--fit-replicates",
        "--decision-replicates",
        "--batch-size",
        "--d-model",
        "--hidden-size",
        "--route-threshold",
        "--student-",
        "--lr",
        "--steps",
        "--temperature",
        "--threshold",
        "--class-weight",
    ):
        assert forbidden not in help_text


def test_v9_reducer_clis_have_only_source_receipts_and_output() -> None:
    budget_help = subprocess.run(
        [sys.executable, str(BUDGET_SCRIPT), "--help"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.lower()
    cross_help = subprocess.run(
        [sys.executable, str(CROSS_SCRIPT), "--help"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.lower()
    assert "--shard" in budget_help and "--output" in budget_help
    assert "--train60" in cross_help and "--train120" in cross_help and "--output" in cross_help
    for text in (budget_help, cross_help):
        for forbidden in ("--classification", "--decision", "--authorize", "--threshold", "--tune"):
            assert forbidden not in text


def test_v9_all_clis_write_canonical_bytes_with_sha256_sidecars(tmp_path: Path) -> None:
    receipt = {"b": 2, "a": 1}
    expected = canonical_receipt_bytes(receipt)
    for index, script in enumerate((SHARD_SCRIPT, BUDGET_SCRIPT, CROSS_SCRIPT)):
        module = _load_script(script, f"v9_cli_{index}")
        output = tmp_path / f"receipt-{index}.json"
        module._write_receipt(output, receipt)
        assert output.read_bytes() == expected
        sidecar = output.with_name(output.name + ".sha256")
        assert sidecar.read_text(encoding="utf-8") == hashlib.sha256(expected).hexdigest() + "\n"


def test_v9_shard_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    output = tmp_path / "v9-shard.json"
    output.write_text("sentinel\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SHARD_SCRIPT),
            "--train-replicates", "60",
            "--canonical-index", "0",
            "--scientific-branch-head", "a" * 40,
            "--executed-commit", "b" * 40,
            "--output", str(output),
            "--protocol", str(tmp_path / "missing.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"
