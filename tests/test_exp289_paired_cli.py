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
        str(ROOT / "scripts" / "run_exp289_paired_dev.py"),
        "--tiny",
        "--train-replicates", "2",
        "--eval-replicates", "3",
        "--batch-size", "2",
        "--timesteps", "3",
        "--restarts", "3",
        "--variables", "6",
        "--decoys", "2",
        "--max-search-steps", "12",
        "--output", str(output),
    ]
    if registry is not None:
        command.extend(["--registry-output", str(registry)])
    return command


def test_exp289_paired_cli_writes_valid_development_and_registry_artifacts(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp289_paired_runner import (
        _artifact_digest,
        validate_exp289_execution_artifact,
        validate_exp289_paired_development,
    )
    from nolane_ai.experiments.neural_arm_registry import validate_neural_arm_registry

    execution_path = tmp_path / "exp289.json"
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

    assert execution["schema"] == "NLM-EXP-289-PAIRED-DEV-EVAL-V1"
    assert execution["evidence_level"] == "EV-E2"
    assert execution["decision"] == "UNVERIFIED"
    assert execution["confirmatory_ready"] is False
    assert execution["confirmatory_data_consumed"] is False
    assert execution["challenge_seed_materialized"] is False
    assert execution["challenge_materialized"] is False
    assert execution["decision_rule_executed"] is False
    assert execution["learned_clause_generalization_validated"] is False
    assert execution["cross_episode_memory_reuse_validated"] is False
    assert execution["artifact_digest"] == _artifact_digest(execution)
    assert validate_exp289_paired_development(execution) == []
    validate_exp289_execution_artifact(execution, execution["frozen_contract"])

    exp289 = registry["experiments"]["EXP-289"]
    assert exp289["development_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
    assert exp289["development_match_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
    assert exp289["match_court"] == "BLOCKED"
    paired = exp289["paired_execution_evidence"]
    assert paired["artifact_digest"] == execution["artifact_digest"]
    assert paired["learned_clause_transfer_validated"] is False
    assert paired["lifelong_lemma_economy_validated"] is False
    assert validate_neural_arm_registry(registry) == []

    assert stdout["execution_digest"] == execution["artifact_digest"]
    assert stdout["registry_digest"] == registry["registry_digest"]
    assert stdout["match_court"] == "BLOCKED"
    assert stdout["development_match_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
    assert stdout["confirmatory_data_consumed"] is False
    assert stdout["challenge_materialized"] is False
    assert stdout["learned_clause_transfer_validated"] is False
    assert stdout["lifelong_lemma_economy_validated"] is False


def test_exp289_paired_cli_supports_execution_only_output(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp289_paired_runner import validate_exp289_paired_development

    execution_path = tmp_path / "exp289.json"
    completed = subprocess.run(
        _base_command(execution_path),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert validate_exp289_paired_development(execution) == []
    assert stdout["registry_digest"] is None
    assert stdout["match_court"] is None


def test_exp289_paired_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp289.json"
    registry_path = tmp_path / "registry.json"
    execution_path.write_text("sentinel\n", encoding="utf-8")
    command = _base_command(execution_path, registry_path)
    command.extend(["--protocol", str(tmp_path / "missing-protocol.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert execution_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not registry_path.exists()


def test_exp289_paired_cli_refuses_existing_registry_before_protocol_work(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp289.json"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("sentinel\n", encoding="utf-8")
    command = _base_command(execution_path, registry_path)
    command.extend(["--protocol", str(tmp_path / "missing-protocol.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert registry_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not execution_path.exists()


def test_exp289_paired_cli_rejects_noncanonical_protocol_even_with_matching_digest(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp289.json"
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


def test_exp289_paired_cli_failure_after_protocol_validation_leaves_no_partial_outputs(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp289.json"
    registry_path = tmp_path / "registry.json"
    command = _base_command(execution_path, registry_path)
    command.extend(["--eval-start-replicate", "1"])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "disjoint" in completed.stderr.lower()
    assert not execution_path.exists()
    assert not registry_path.exists()
