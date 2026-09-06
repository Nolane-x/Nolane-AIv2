from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.identity import source_tree_digest
from tests.exp282_confirmatory_fixtures import prepared_chain

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_exp282_confirmatory_open.py"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _command(tmp_path: Path, output: Path) -> list[str]:
    analysis_digest = source_tree_digest(ROOT)
    _, prep, execution, reconstruction, raw = prepared_chain(
        tmp_path,
        analysis_code_digest=analysis_digest,
    )
    prep_path = tmp_path / "prep.json"
    execution_path = tmp_path / "paired.json"
    reconstruction_path = tmp_path / "reconstruction.json"
    raw_path = tmp_path / "raw.json"
    _write_json(prep_path, prep)
    _write_json(execution_path, execution)
    _write_json(reconstruction_path, reconstruction)
    _write_json(raw_path, raw)
    return [
        sys.executable,
        str(SCRIPT),
        "--protocol",
        str(ROOT / "protocols" / "stage_a_v1.json"),
        "--protocol-digest-file",
        str(ROOT / "protocols" / "stage_a_v1.sha256"),
        "--prep",
        str(prep_path),
        "--execution",
        str(execution_path),
        "--reconstruction",
        str(reconstruction_path),
        "--raw",
        str(raw_path),
        "--output",
        str(output),
    ]


def test_confirmatory_analysis_cli_writes_self_validating_ev_e3_result(tmp_path: Path):
    from nolane_ai.experiments.exp282_confirmatory_analysis import validate_exp282_confirmatory_analysis

    output = tmp_path / "analysis.json"
    completed = subprocess.run(_command(tmp_path, output), cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == "NLM-EXP-282-CONFIRMATORY-ANALYSIS-V1"
    assert result["evidence_level"] == "EV-E3"
    assert result["decision"] in {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
    assert validate_exp282_confirmatory_analysis(result) == []
    summary = json.loads(completed.stdout)
    assert summary["schema"] == result["schema"]
    assert summary["evidence_level"] == "EV-E3"
    assert summary["decision"] == result["decision"]
    assert summary["analysis_digest"] == result["analysis_digest"]
    assert summary["brier_guard_pass"] == result["protected_endpoints"]["brier"]["pass"]
    assert summary["compute_guard_pass"] == result["protected_endpoints"]["compute"]["pass"]


def test_confirmatory_analysis_cli_refuses_overwrite_before_analysis(tmp_path: Path):
    output = tmp_path / "analysis.json"
    output.write_text("sentinel\n", encoding="utf-8")
    completed = subprocess.run(_command(tmp_path, output), cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "output already exists" in (completed.stderr + completed.stdout)
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_confirmatory_analysis_cli_rejects_forged_protocol_and_matching_digest_file(tmp_path: Path):
    output = tmp_path / "analysis.json"
    command = _command(tmp_path, output)
    protocol_index = command.index("--protocol") + 1
    digest_index = command.index("--protocol-digest-file") + 1
    canonical_protocol = Path(command[protocol_index])
    forged = json.loads(canonical_protocol.read_text(encoding="utf-8"))
    forged["rng"]["root_seed"] = "forged-analysis-root"
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
