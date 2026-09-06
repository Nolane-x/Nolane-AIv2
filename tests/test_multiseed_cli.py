import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def test_multiseed_cli_writes_ev_e2_artifact(tmp_path: Path):
    output = tmp_path / "multi.json"
    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "run_multiseed_heldout.py"),
        "--tiny",
        "--seeds", "2",
        "--train-steps", "1",
        "--eval-batches", "1",
        "--batch-size", "2",
        "--variables", "2",
        "--constraints", "2",
        "--bootstrap-samples", "50",
        "--output", str(output),
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert payload["schema"] == "NLM-STAGE-A-MULTISEED-HELDOUT-DEV-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["seed_count"] == 2
    assert payload["successful_seed_count"] == 2
    assert payload["failed_seed_count"] == 0
    assert payload["aggregate_valid"] is True
    assert len(payload["per_seed"]) == 2
    assert stdout["multiseed_digest"] == payload["multiseed_digest"]
    assert stdout["aggregate_valid"] is True
