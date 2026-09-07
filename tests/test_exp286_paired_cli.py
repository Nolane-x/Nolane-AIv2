from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _base_command(output: Path, registry: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_exp286_paired_dev.py"),
        "--tiny",
        "--train-replicates", "2",
        "--eval-replicates", "3",
        "--batch-size", "2",
        "--timesteps", "3",
        "--variables", "5",
        "--decoys", "2",
        "--max-search-steps", "8",
        "--output", str(output),
        "--registry-output", str(registry),
    ]


def test_exp286_paired_cli_writes_development_and_registry_artifacts(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp286.json"
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

    assert execution["schema"] == "NLM-EXP-286-PAIRED-DEV-EVAL-V1"
    assert execution["evidence_level"] == "EV-E2"
    assert execution["decision"] == "UNVERIFIED"
    assert execution["confirmatory_ready"] is False
    assert execution["confirmatory_data_consumed"] is False
    assert execution["challenge_seed_materialized"] is False
    assert execution["challenge_materialized"] is False
    assert execution["decision_rule_executed"] is False
    exp286 = registry["experiments"]["EXP-286"]
    assert exp286["development_match_status"] == "PAIRED_CONFLICT_HEADROOM_DEV_READY"
    assert exp286["match_court"] == "BLOCKED"
    evidence = exp286["paired_execution_evidence"]
    assert evidence["artifact_digest"] == execution["artifact_digest"]
    assert evidence["learned_conflict_localizer_validated"] is False
    assert evidence["chronological_censored_episode_count"] == execution["evaluation"]["aggregate"][
        "chronological_censored_episode_count"
    ]
    assert evidence["oracle_censored_episode_count"] == execution["evaluation"]["aggregate"][
        "oracle_censored_episode_count"
    ]
    assert stdout["execution_digest"] == execution["artifact_digest"]
    assert stdout["registry_digest"] == registry["registry_digest"]
    assert stdout["match_court"] == "BLOCKED"
    assert stdout["development_match_status"] == "PAIRED_CONFLICT_HEADROOM_DEV_READY"
    assert stdout["confirmatory_data_consumed"] is False
    assert stdout["challenge_materialized"] is False
    assert stdout["learned_conflict_localizer_validated"] is False


def test_exp286_paired_cli_refuses_existing_output_before_protocol_work(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp286.json"
    registry_path = tmp_path / "registry.json"
    execution_path.write_text("sentinel\n", encoding="utf-8")
    command = _base_command(execution_path, registry_path)
    command.extend(["--protocol", str(tmp_path / "missing-protocol.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert execution_path.read_text(encoding="utf-8") == "sentinel\n"
    assert not registry_path.exists()


def test_exp286_paired_cli_rejects_noncanonical_protocol_even_with_matching_digest(tmp_path: Path) -> None:
    execution_path = tmp_path / "exp286.json"
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
