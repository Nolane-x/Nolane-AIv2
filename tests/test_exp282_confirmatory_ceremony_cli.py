from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp282_confirmatory_ceremony import seal_exp282_confirmatory_ceremony
from nolane_ai.experiments.exp282_confirmatory_execution_court import authorize_exp282_confirmatory_execution
from nolane_ai.experiments.exp282_reconstruction_court import authorize_exp282_confirmatory_reconstruction
from nolane_ai.protocol.identity import source_tree_digest
from tests.exp282_confirmatory_fixtures import CANONICAL_PROTOCOL_DIGEST, prepared_chain

ROOT = Path(__file__).resolve().parents[1]
SEAL_SCRIPT = ROOT / "scripts" / "seal_exp282_confirmatory_ceremony.py"
EXECUTE_SCRIPT = ROOT / "scripts" / "execute_exp282_confirmatory_ceremony.py"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _artifacts(tmp_path: Path):
    digest = source_tree_digest(ROOT)
    protocol, prep, execution, _, _ = prepared_chain(tmp_path, analysis_code_digest=digest)
    authorization = authorize_exp282_confirmatory_execution(
        prep_artifact=prep,
        execution_code_digest=digest,
    )
    reconstruction = authorize_exp282_confirmatory_reconstruction(
        execution_artifact=execution,
        execution_authorization=authorization,
    )
    paths = {
        "execution": tmp_path / "execution.json",
        "prep": tmp_path / "prep.json",
    }
    _write_json(paths["execution"], execution)
    _write_json(paths["prep"], prep)
    return digest, protocol, prep, execution, authorization, reconstruction, paths


def _seal_command(tmp_path: Path, output: Path) -> list[str]:
    _, _, _, _, _, _, paths = _artifacts(tmp_path)
    return [
        sys.executable,
        str(SEAL_SCRIPT),
        "--protocol",
        str(ROOT / "protocols" / "stage_a_v1.json"),
        "--protocol-digest-file",
        str(ROOT / "protocols" / "stage_a_v1.sha256"),
        "--execution",
        str(paths["execution"]),
        "--prep",
        str(paths["prep"]),
        "--output",
        str(output),
    ]


def _execute_inputs(tmp_path: Path):
    digest, protocol, prep, execution, authorization, reconstruction, paths = _artifacts(tmp_path)
    seal = seal_exp282_confirmatory_ceremony(
        protocol_digest=CANONICAL_PROTOCOL_DIGEST,
        paired_execution_artifact=execution,
        prep_artifact=prep,
        execution_authorization=authorization,
        reconstruction_authorization=reconstruction,
        ceremony_code_digest=digest,
    )
    seal_path = tmp_path / "seal.json"
    _write_json(seal_path, seal)
    return seal_path, paths["execution"], tmp_path / "paired.pt"


def test_seal_cli_writes_self_validating_unexecuted_seal(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_ceremony import validate_exp282_confirmatory_ceremony_seal

    output = tmp_path / "seal-output.json"
    completed = subprocess.run(_seal_command(tmp_path, output), cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    seal = json.loads(output.read_text(encoding="utf-8"))
    assert seal["status"] == "CEREMONY_SEALED_NOT_EXECUTED"
    assert seal["confirmatory_data_consumed"] is False
    assert seal["challenge_materialized"] is False
    assert seal["authorities"]["execution_authorization"]["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert seal["authorities"]["reconstruction_authorization"]["status"] == "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
    assert validate_exp282_confirmatory_ceremony_seal(seal) == []


def test_seal_cli_refuses_overwrite_before_sealing(tmp_path: Path):
    output = tmp_path / "seal-output.json"
    output.write_text("sentinel\n", encoding="utf-8")
    completed = subprocess.run(_seal_command(tmp_path, output), cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in (completed.stderr + completed.stdout)
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_seal_cli_rejects_forged_protocol_and_matching_digest(tmp_path: Path):
    output = tmp_path / "seal-output.json"
    command = _seal_command(tmp_path, output)
    protocol_index = command.index("--protocol") + 1
    digest_index = command.index("--protocol-digest-file") + 1
    canonical = Path(command[protocol_index])
    forged = json.loads(canonical.read_text(encoding="utf-8"))
    forged["rng"]["root_seed"] = "forged-ceremony-root"
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    forged_digest = hashlib.sha256(forged_path.read_bytes()).hexdigest()
    forged_digest_path = tmp_path / "forged.sha256"
    forged_digest_path.write_text(forged_digest + "\n", encoding="utf-8")
    command[protocol_index] = str(forged_path)
    command[digest_index] = str(forged_digest_path)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "canonical frozen protocol digest" in (completed.stderr + completed.stdout)
    assert not output.exists()


def test_execute_cli_writes_raw_analysis_and_bundle(tmp_path: Path):
    seal_path, execution_path, checkpoint_path = _execute_inputs(tmp_path)
    raw_path = tmp_path / "raw.json"
    analysis_path = tmp_path / "analysis.json"
    bundle_path = tmp_path / "bundle.json"
    command = [
        sys.executable,
        str(EXECUTE_SCRIPT),
        "--protocol",
        str(ROOT / "protocols" / "stage_a_v1.json"),
        "--protocol-digest-file",
        str(ROOT / "protocols" / "stage_a_v1.sha256"),
        "--seal",
        str(seal_path),
        "--execution",
        str(execution_path),
        "--checkpoint",
        str(checkpoint_path),
        "--raw-output",
        str(raw_path),
        "--analysis-output",
        str(analysis_path),
        "--bundle-output",
        str(bundle_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert raw["evidence_level"] == "EV-E2"
    assert analysis["evidence_level"] == "EV-E3"
    assert bundle["lineage"]["raw_artifact_digest"] == raw["artifact_digest"]
    assert bundle["lineage"]["analysis_digest"] == analysis["analysis_digest"]
    assert bundle["challenge_materialized"] is False


def test_execute_cli_refuses_any_output_overwrite_before_execution(tmp_path: Path):
    seal_path, execution_path, checkpoint_path = _execute_inputs(tmp_path)
    raw_path = tmp_path / "raw.json"
    analysis_path = tmp_path / "analysis.json"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("sentinel\n", encoding="utf-8")
    command = [
        sys.executable,
        str(EXECUTE_SCRIPT),
        "--seal",
        str(seal_path),
        "--execution",
        str(execution_path),
        "--checkpoint",
        str(checkpoint_path),
        "--raw-output",
        str(raw_path),
        "--analysis-output",
        str(analysis_path),
        "--bundle-output",
        str(bundle_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in (completed.stderr + completed.stdout)
    assert not raw_path.exists()
    assert not analysis_path.exists()
    assert bundle_path.read_text(encoding="utf-8") == "sentinel\n"
