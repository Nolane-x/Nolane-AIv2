from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_audit_script_reports_exact_16m():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_stage_a_pilot.py")],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["total_parameters"] == 16_000_000
    assert payload["budget_version"] == "stage-a-pilot-16m-v1"
    assert payload["device"] == "meta"


def test_tiny_pilot_training_cli_writes_ev_e2_checkpoint_manifest(tmp_path: Path):
    output_dir = tmp_path / "pilot"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_stage_a_pilot.py"),
            "--tiny",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--variables",
            "3",
            "--constraints",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    manifest_path = output_dir / "pilot.manifest.json"
    tensor_path = output_dir / "pilot.pt"
    assert manifest_path.exists() and tensor_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_level"] == "EV-E2"
    assert manifest["decision"] == "UNVERIFIED"
    assert manifest["training_step"] == 1
    assert manifest["protocol_digest"]
    assert manifest["code_digest"]
