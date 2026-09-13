from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_exp298_cross_domain_fidelity_dev.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("exp298_cli", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_identity_modes_but_no_scientific_tuning_flags():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = completed.stdout
    assert "root" in text and "cross" in text
    for forbidden in (
        "--eval-replicates",
        "--eval-start-replicate",
        "--d-model",
        "--hidden-size",
        "--target-parameters",
        "--max-exact-probes",
        "--mesi",
        "--threshold",
    ):
        assert forbidden not in text


def test_root_parser_requires_only_canonical_index_and_output(monkeypatch, tmp_path):
    module = _load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "root", "--canonical-index", "2", "--output", str(tmp_path / "root.json")],
    )
    args = module.parse_args()
    assert args.mode == "root"
    assert args.canonical_index == 2
    assert args.output == tmp_path / "root.json"


def test_cross_parser_requires_exactly_four_root_receipts(monkeypatch, tmp_path):
    module = _load_script()
    roots = [tmp_path / f"root-{i}.json" for i in range(4)]
    argv = [str(SCRIPT), "cross"]
    for root in roots:
        argv.extend(["--root-receipt", str(root)])
    argv.extend(["--output", str(tmp_path / "cross.json")])
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    assert args.mode == "cross"
    assert args.root_receipt == roots


def test_canonical_publication_is_atomic_has_sidecar_and_refuses_overwrite(tmp_path):
    module = _load_script()
    output = tmp_path / "receipt.json"
    receipt = {"z": 2, "a": {"b": 1}}
    module._write_canonical_atomic(output, receipt)
    expected = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    assert output.read_bytes() == expected
    assert output.with_name(output.name + ".sha256").read_text() == hashlib.sha256(expected).hexdigest() + "\n"
    with pytest.raises(FileExistsError):
        module._write_canonical_atomic(output, receipt)
