from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _base_command(output: Path, registry: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_exp297_paired_dev.py"),
        "--tiny",
        "--eval-replicates", "1",
        "--eval-start-replicate", "1000",
        "--output", str(output),
    ]
    if registry is not None:
        command.extend(["--registry-output", str(registry)])
    return command


def test_exp297_cli_writes_valid_execution_and_extended_registry(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp297_paired_runner import validate_exp297_execution
    from nolane_ai.experiments.exp297_registry import validate_exp297_neural_arm_registry

    execution_path = tmp_path / "exp297.json"
    registry_path = tmp_path / "registry.json"
    completed = subprocess.run(
        _base_command(execution_path, registry_path),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())

    assert validate_exp297_execution(execution) == []
    assert validate_exp297_neural_arm_registry(registry) == []
    item = registry["experiments"]["EXP-297"]
    assert item["development_match_status"] == "PAIRED_FIDELITY_COURT_DEV_READY"
    assert item["match_court"] == "BLOCKED"
    assert item["paired_execution_evidence"]["execution_digest"] == execution["artifact_digest"]
    assert item["paired_execution_evidence"]["semantic_authority_promoted"] is False
    assert stdout["execution_digest"] == execution["artifact_digest"]
    assert stdout["registry_digest"] == registry["registry_digest"]
    assert stdout["match_court"] == "BLOCKED"
    assert stdout["semantic_authority_promoted"] is False


def test_exp297_cli_supports_execution_only_output(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp297.json"
    completed = subprocess.run(
        _base_command(execution_path),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = json.loads(completed.stdout.strip())
    assert execution_path.exists()
    assert stdout["registry_digest"] is None
    assert stdout["match_court"] is None


def test_exp297_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp297.json"
    registry_path = tmp_path / "registry.json"
    execution_path.write_text("sentinel\n", encoding="utf-8")
    command = _base_command(execution_path, registry_path)
    command.extend(["--protocol", str(tmp_path / "missing.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert execution_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not registry_path.exists()


def test_exp297_cli_rejects_noncanonical_protocol_even_with_matching_digest(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp297.json"
    registry_path = tmp_path / "registry.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes((ROOT / "protocols" / "stage_a_v1.json").read_bytes() + b"\n")
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    command = _base_command(execution_path, registry_path)
    command.extend([
        "--protocol", str(protocol),
        "--protocol-digest-file", str(digest),
    ])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "canonical frozen protocol digest mismatch" in completed.stderr
    assert not execution_path.exists()
    assert not registry_path.exists()


def test_exp297_cli_failure_after_protocol_validation_leaves_no_partial_outputs(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp297.json"
    registry_path = tmp_path / "registry.json"
    command = _base_command(execution_path, registry_path)
    command.extend(["--max-exact-assignments", "0"])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert completed.returncode != 0
    assert not execution_path.exists()
    assert not registry_path.exists()
