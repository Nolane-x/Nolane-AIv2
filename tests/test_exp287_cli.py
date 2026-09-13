from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp287_receipts import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
ROOT_CLI = ROOT / "scripts" / "run_exp287_learned_conflict_localization_dev.py"
CROSS_CLI = ROOT / "scripts" / "reduce_exp287_learned_conflict_localization_dev.py"

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
    "--max-search-steps",
    "--noise-std",
    "--lr",
    "--weight-decay",
    "--top-k",
    "--capture-threshold",
    "--precision-threshold",
    "--solution-rate-floor",
    "--loss-weight",
    "--class-weight",
    "--temperature",
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
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp287_root_cli", ROOT_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "receipt.json"
    receipt = {"schema": "test", "value": 1}
    module._write_receipt_atomic(output, receipt)
    payload = canonical_json_bytes(receipt)
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
