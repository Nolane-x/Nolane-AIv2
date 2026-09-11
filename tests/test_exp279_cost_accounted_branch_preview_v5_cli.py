from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_exp279_cost_accounted_branch_preview_v5_dev.py"


def _command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--tiny",
        "--train-replicates", "3",
        "--probe-replicates", "9",
        "--probe-folds", "3",
        "--probe-steps", "1",
        "--batch-size", "2",
        "--timesteps", "4",
        "--variables", "4",
        "--constraints", "2",
        "--route-threshold", "0.5",
        "--output", str(output),
    ]


def test_v5_cli_writes_cost_accounted_augmentation_only_artifact(tmp_path: Path) -> None:
    output = tmp_path / "preview-v5.json"
    completed = subprocess.run(_command(output), cwd=ROOT, check=True, text=True, capture_output=True)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert artifact["schema"] == "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-COURT-V5"
    assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
    assert artifact["data_boundary"]["evaluation_targets_used"] is False
    assert artifact["preview_cost_rule"]["preview_flops_charged_to_primary_utility"] is True
    assert artifact["preview_cost_rule"]["selector_flops_charged_to_primary_utility"] is True
    assert artifact["fresh_evaluation_lineage_may_be_reserved"] is False
    assert artifact["fresh_evaluation_lineage_consumed"] is False
    assert stdout["artifact_digest"] == artifact["artifact_digest"]


def test_v5_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    output = tmp_path / "preview-v5.json"
    output.write_text("sentinel\n", encoding="utf-8")
    command = _command(output) + ["--protocol", str(tmp_path / "missing.json")]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_v5_cli_rejects_noncanonical_protocol_even_with_matching_digest(tmp_path: Path) -> None:
    output = tmp_path / "preview-v5.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes((ROOT / "protocols" / "stage_a_v1.json").read_bytes() + b"\n")
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    command = _command(output) + [
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "canonical frozen protocol digest mismatch" in completed.stderr
    assert not output.exists()
