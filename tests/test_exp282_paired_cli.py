import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _write_protocol_fixture(tmp_path: Path):
    source = json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))
    expected = {
        "EXP-277": [
            {"id": "arcs_branch", "description": "V0.15 ARCS recurrent-depth + branch bank + verifier court"},
            {"id": "oracle_cbrf", "description": "same substrate with ground-truth constraint/factor representation"},
        ],
        "EXP-279": [
            {"id": "propagation_only", "description": "constraint propagation without branch search"},
            {"id": "branch_only", "description": "ARCS branch search without CBRF propagation"},
            {"id": "hybrid", "description": "propagation followed by branch search when residual uncertainty remains"},
        ],
        "EXP-282": [
            {"id": "recurrent_hidden", "description": "matched recurrent state without explicit belief representation"},
            {"id": "explicit_belief", "description": "explicit calibrated belief state over hidden world variables"},
        ],
    }
    resource = {
        "EXP-277": {"parameter_budget": "matched active parameter count", "inference_budget": "same max accounted FLOPs per episode", "world_pairing": "same world lineage and replicate index"},
        "EXP-279": {"parameter_budget": "reclaimed parameters assigned to simpler rivals", "inference_budget": "equal max accounted FLOPs", "structure_fit": "predeclared strata"},
        "EXP-282": {"parameter_budget": "equal state/controller parameters", "observation_history": "identical", "compute_budget": "matched"},
    }
    for experiment in source["experiments"]:
        if experiment["experiment_id"] in expected:
            experiment["arms"] = expected[experiment["experiment_id"]]
            experiment["resource_match"] = resource[experiment["experiment_id"]]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = tmp_path / "protocol.sha256"
    digest.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    return protocol, digest


def test_exp282_paired_cli_writes_execution_and_match_court_artifacts(tmp_path: Path):
    execution = tmp_path / "exp282.json"
    registry = tmp_path / "registry.json"
    protocol, protocol_digest = _write_protocol_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_exp282_paired_dev.py"),
            "--tiny",
            "--protocol", str(protocol),
            "--protocol-digest-file", str(protocol_digest),
            "--train-replicates", "2",
            "--eval-replicates", "2",
            "--batch-size", "2",
            "--timesteps", "3",
            "--variables", "2",
            "--output", str(execution),
            "--registry-output", str(registry),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    run = json.loads(execution.read_text(encoding="utf-8"))
    court = json.loads(registry.read_text(encoding="utf-8"))
    stdout = json.loads(completed.stdout.strip())
    assert run["schema"] == "NLM-EXP-282-PAIRED-DEV-EVAL-V1"
    assert run["evidence_level"] == "EV-E2"
    assert run["decision"] == "UNVERIFIED"
    assert court["schema"] == "NLM-STAGE-A-NEURAL-ARM-REGISTRY-V1"
    exp282 = court["experiments"]["EXP-282"]
    assert exp282["development_match_status"] == "PAIRED_PARTIAL_OBSERVABILITY_DEV_READY"
    assert exp282["match_court"] == "BLOCKED"
    assert exp282["paired_execution_evidence"]["artifact_digest"] == run["artifact_digest"]
    assert stdout["execution_digest"] == run["artifact_digest"]
    assert stdout["registry_digest"] == court["registry_digest"]
