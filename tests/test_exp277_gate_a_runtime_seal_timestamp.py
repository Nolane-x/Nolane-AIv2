from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "scripts" / "prepare_exp277_confirmatory_gate_a.py"
GATE_A_TESTS = ROOT / "tests" / "test_exp277_confirmatory_gate_a_cli.py"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp277_gate_a_captures_checkpoint_seal_timestamp_after_checkpoint_creation(
    tmp_path: Path,
) -> None:
    helpers = _load_script(GATE_A_TESTS, "exp277_gate_a_test_helpers")
    execution, registry = helpers._make_stable_development()
    execution_path = tmp_path / "execution.json"
    registry_path = tmp_path / "registry.json"
    helpers._write_json(execution_path, execution)
    helpers._write_json(registry_path, registry)

    args, paths = helpers._gate_a_args(tmp_path / "runtime-seal", execution_path, registry_path)
    flag_index = args.index("--checkpoint-seal-created-at-utc")
    del args[flag_index : flag_index + 2]

    gate_a = _load_script(GATE_A, "exp277_gate_a_runtime_seal")
    before = datetime.now(timezone.utc)
    assert gate_a.main(args) == 0
    after = datetime.now(timezone.utc)

    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    recorded_text = authorization["checkpoint_seal_created_at_utc"]
    recorded = datetime.fromisoformat(recorded_text.replace("Z", "+00:00"))
    checkpoint_written = datetime.fromtimestamp(
        paths["checkpoint"].stat().st_mtime,
        timezone.utc,
    )

    assert before <= recorded <= after
    assert recorded >= checkpoint_written
    assert authorization["freeze_commit_timestamp_utc"] < recorded_text
