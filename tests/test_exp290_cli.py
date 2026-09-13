from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp290_receipts import canonical_receipt_bytes


ROOT = Path(__file__).resolve().parents[1]
ROOT_CLI = ROOT / "scripts" / "run_exp290_structural_clause_transfer_dev.py"
CROSS_CLI = ROOT / "scripts" / "reduce_exp290_structural_clause_transfer_dev.py"

FORBIDDEN_TUNING_FLAGS = (
    "--train-replicates",
    "--eval-replicates",
    "--eval-start-replicate",
    "--batch-size",
    "--d-model",
    "--hidden-size",
    "--target-parameters",
    "--timesteps",
    "--variables",
    "--decoys",
    "--restarts",
    "--max-search-steps",
    "--noise-std",
    "--lr",
    "--weight-decay",
    "--top-k",
    "--capture-threshold",
    "--overprune-threshold",
    "--solution-rate-floor",
    "--loss-weight",
    "--class-weight",
    "--temperature",
    "--surface-randomization",
    "--permutation",
    "--root-prefix",
)


def _help(path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _load_cli(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_cli_exposes_only_canonical_index_and_output_as_variable_surface() -> None:
    text = _help(ROOT_CLI)
    assert "--canonical-index" in text
    assert "--output" in text
    for flag in FORBIDDEN_TUNING_FLAGS:
        assert flag not in text


def test_cross_cli_requires_exactly_four_root_receipts_and_output() -> None:
    text = _help(CROSS_CLI)
    assert "--root-receipts" in text
    assert "--output" in text
    for flag in FORBIDDEN_TUNING_FLAGS:
        assert flag not in text

    completed = subprocess.run(
        [
            sys.executable,
            str(CROSS_CLI),
            "--root-receipts",
            "a.json",
            "b.json",
            "c.json",
            "--output",
            "cross.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0


def test_atomic_receipt_writer_refuses_overwrite_and_writes_matching_sidecar(tmp_path: Path) -> None:
    module = _load_cli(ROOT_CLI, "exp290_root_cli")
    output = tmp_path / "receipt.json"
    receipt = {"schema": "test", "value": 1}

    module._write_receipt_atomic(output, receipt)
    payload = canonical_receipt_bytes(receipt)
    assert output.read_bytes() == payload
    assert output.with_name("receipt.json.sha256").read_text(encoding="utf-8") == (
        hashlib.sha256(payload).hexdigest() + "\n"
    )

    with pytest.raises(FileExistsError, match="already exists"):
        module._write_receipt_atomic(output, receipt)


def test_cross_cli_rejects_non_object_json_before_publication(tmp_path: Path) -> None:
    roots = []
    for index in range(4):
        path = tmp_path / f"root-{index}.json"
        path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        roots.append(path)
    output = tmp_path / "cross.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CROSS_CLI),
            "--root-receipts",
            *(str(path) for path in roots),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert not output.with_name("cross.json.sha256").exists()


def test_root_cli_source_binds_frozen_geometry_and_source_tree_digest() -> None:
    source = ROOT_CLI.read_text(encoding="utf-8")
    assert "load_exp290_development_geometry" in source
    assert "source_tree_digest(ROOT)" in source
    assert "run_exp290_root" in source
    assert "seal_root_receipt" in source
    assert "validate_root_receipt" in source


def test_cross_cli_validates_each_root_and_cross_before_publication() -> None:
    source = CROSS_CLI.read_text(encoding="utf-8")
    assert "validate_root_receipt" in source
    assert "reduce_cross_roots" in source
    assert "validate_cross_receipt" in source
