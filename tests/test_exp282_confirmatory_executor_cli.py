import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _helper_module():
    helper_path = ROOT / "tests" / "test_exp282_confirmatory_executor.py"
    spec = importlib.util.spec_from_file_location("exp282_executor_contract_test_helper", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path):
    protocol_path = ROOT / "protocols" / "stage_a_v1.json"
    digest_path = ROOT / "protocols" / "stage_a_v1.sha256"
    protocol_digest = digest_path.read_text(encoding="utf-8").strip()
    _, _, execution, reconstruction, checkpoint_path = _helper_module()._prepared_chain(
        tmp_path,
        protocol_digest=protocol_digest,
    )
    execution_path = tmp_path / "paired.json"
    reconstruction_path = tmp_path / "reconstruction.json"
    execution_path.write_text(json.dumps(execution, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    reconstruction_path.write_text(json.dumps(reconstruction, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return protocol_path, digest_path, execution_path, reconstruction_path, checkpoint_path


def _command(tmp_path: Path, output: Path):
    protocol_path, digest_path, execution_path, reconstruction_path, checkpoint_path = _write_inputs(tmp_path)
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_exp282_confirmatory_open.py"),
        "--protocol", str(protocol_path),
        "--protocol-digest-file", str(digest_path),
        "--execution", str(execution_path),
        "--reconstruction", str(reconstruction_path),
        "--checkpoint", str(checkpoint_path),
        "--output", str(output),
    ], checkpoint_path


def test_confirmatory_executor_cli_writes_raw_ev_e2_artifact(tmp_path: Path):
    output = tmp_path / "confirmatory-raw.json"
    command, _ = _command(tmp_path, output)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(completed.stdout.strip())
    assert payload["schema"] == "NLM-EXP-282-CONFIRMATORY-OPEN-RAW-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["status"] == "CONFIRMATORY_OPEN_EXECUTED_UNANALYZED"
    assert payload["confirmatory_n"] == 32
    assert summary == {
        "artifact_digest": payload["artifact_digest"],
        "confirmatory_n": 32,
        "schema": payload["schema"],
        "status": payload["status"],
    }


def test_confirmatory_executor_cli_refuses_to_overwrite_existing_output(tmp_path: Path):
    output = tmp_path / "confirmatory-raw.json"
    output.write_text("do-not-overwrite\n", encoding="utf-8")
    command, _ = _command(tmp_path, output)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in (completed.stderr + completed.stdout)
    assert output.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_confirmatory_executor_cli_rejects_checkpoint_sha_mismatch(tmp_path: Path):
    output = tmp_path / "confirmatory-raw.json"
    command, checkpoint_path = _command(tmp_path, output)
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "checkpoint SHA mismatch" in (completed.stderr + completed.stdout)
    assert not output.exists()


def test_confirmatory_executor_cli_rejects_forged_protocol_and_matching_digest_file(tmp_path: Path):
    output = tmp_path / "confirmatory-raw.json"
    command, _ = _command(tmp_path, output)
    protocol_index = command.index("--protocol") + 1
    digest_index = command.index("--protocol-digest-file") + 1
    canonical_protocol = Path(command[protocol_index])
    forged = json.loads(canonical_protocol.read_text(encoding="utf-8"))
    forged["rng"]["root_seed"] = "forged-root"
    forged_protocol = tmp_path / "forged-protocol.json"
    forged_protocol.write_text(json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    forged_digest = hashlib.sha256(forged_protocol.read_bytes()).hexdigest()
    forged_digest_path = tmp_path / "forged-protocol.sha256"
    forged_digest_path.write_text(forged_digest + "\n", encoding="utf-8")
    command[protocol_index] = str(forged_protocol)
    command[digest_index] = str(forged_digest_path)

    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "canonical frozen protocol digest" in (completed.stderr + completed.stdout)
    assert not output.exists()
