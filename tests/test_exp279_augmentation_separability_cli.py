from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _base_command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_exp279_augmentation_separability_dev.py"),
        "--tiny",
        "--train-replicates", "3",
        "--probe-replicates", "9",
        "--probe-folds", "3",
        "--probe-steps", "1",
        "--batch-size", "2",
        "--timesteps", "3",
        "--variables", "4",
        "--constraints", "2",
        "--route-threshold", "0.5",
        "--output", str(output),
    ]


def test_separability_cli_writes_augmentation_only_court_artifact(tmp_path: Path) -> None:
    output = tmp_path / "separability.json"
    completed = subprocess.run(
        _base_command(output),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())

    assert artifact["schema"] == "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
    assert artifact["data_boundary"]["evaluation_targets_used"] is False
    assert artifact["fresh_evaluation_lineage_consumed"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["promotion_claimed"] is False
    assert stdout["artifact_digest"] == artifact["artifact_digest"]
    assert stdout["court_classification"] == artifact["court_classification"]
    assert stdout["fresh_evaluation_lineage_may_be_reserved"] == artifact[
        "fresh_evaluation_lineage_may_be_reserved"
    ]


def test_separability_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    output = tmp_path / "separability.json"
    output.write_text("sentinel\n", encoding="utf-8")
    command = _base_command(output)
    command.extend(["--protocol", str(tmp_path / "missing-protocol.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_separability_cli_rejects_noncanonical_protocol_even_with_matching_digest(tmp_path: Path) -> None:
    output = tmp_path / "separability.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes((ROOT / "protocols" / "stage_a_v1.json").read_bytes() + b"\n")
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    command = _base_command(output)
    command.extend([
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest),
    ])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "canonical frozen protocol digest mismatch" in completed.stderr
    assert not output.exists()
