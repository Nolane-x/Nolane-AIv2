from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "scripts" / "prepare_exp277_confirmatory_gate_a.py"
GATE_B = ROOT / "scripts" / "run_exp277_confirmatory_gate_b.py"
GATE_A_TESTS = ROOT / "tests" / "test_exp277_confirmatory_gate_a_cli.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate_b_inputs(tmp_path_factory: pytest.TempPathFactory):
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt

    helpers = _load_script(GATE_A_TESTS, "exp277_gate_b_persistence_helpers")
    root = tmp_path_factory.mktemp("exp277-gate-b-persistence")
    execution, registry = helpers._make_stable_development()
    execution_path = root / "execution.json"
    registry_path = root / "registry.json"
    helpers._write_json(execution_path, execution)
    helpers._write_json(registry_path, registry)

    gate_a = _load_script(GATE_A, "exp277_gate_a_for_persistence")
    gate_a_args, paths = helpers._gate_a_args(root, execution_path, registry_path)
    assert gate_a.main(gate_a_args) == 0

    beacon_path = root / "beacon.json"
    helpers._write_json(
        beacon_path,
        build_test_beacon_receipt(
            source="synthetic-exp277-raw-persistence",
            beacon_id="test-only-raw-persistence",
            published_at_utc=helpers.BEACON_TIME,
            entropy_hex="56" * 32,
            evidence_reference="test-only://exp277-raw-persistence",
        ),
    )
    return helpers, paths, beacon_path


def _gate_b_args(
    *,
    paths: dict[str, Path],
    beacon_path: Path,
    checkpoint_path: Path,
    raw_path: Path,
    analysis_path: Path,
) -> list[str]:
    return [
        "--protocol", str(PROTOCOL),
        "--protocol-digest-file", str(PROTOCOL_DIGEST),
        "--seal", str(paths["seal"]),
        "--reconstruction", str(paths["reconstruction"]),
        "--beacon", str(beacon_path),
        "--checkpoint", str(checkpoint_path),
        "--checkpoint-receipt", str(paths["checkpoint_receipt"]),
        "--test-only",
        "--raw-output", str(raw_path),
        "--analysis-output", str(analysis_path),
    ]


def test_exp277_gate_b_persists_invalid_raw_instead_of_losing_integrity_evidence(
    gate_b_inputs,
    tmp_path: Path,
) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import validate_exp277_confirmatory_raw

    _, paths, beacon_path = gate_b_inputs
    checkpoint = tmp_path / "tampered-checkpoint.pt"
    shutil.copyfile(paths["checkpoint"], checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    raw_path = tmp_path / "invalid-raw.json"
    analysis_path = tmp_path / "invalid-analysis.json"
    gate_b = _load_script(GATE_B, "exp277_gate_b_invalid_persistence")

    with pytest.raises(RuntimeError, match="INVALID_RUN"):
        gate_b.main(
            _gate_b_args(
                paths=paths,
                beacon_path=beacon_path,
                checkpoint_path=checkpoint,
                raw_path=raw_path,
                analysis_path=analysis_path,
            )
        )

    assert raw_path.exists()
    assert not analysis_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    checkpoint_receipt = json.loads(paths["checkpoint_receipt"].read_text(encoding="utf-8"))
    assert validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=checkpoint,
        checkpoint_receipt=checkpoint_receipt,
    ) == []
    assert raw["status"] == "INVALID_RUN"
    assert raw["decision"] == "INVALID_RUN"
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["challenge_materialized"] is False
    assert raw["decision_rule_executed"] is False
    assert raw["integrity_errors"]


def test_exp277_gate_b_publishes_valid_raw_before_analysis_runs(
    gate_b_inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nolane_ai.experiments.exp277_confirmatory_ceremony as ceremony_module
    from nolane_ai.experiments.exp277_confirmatory_executor import validate_exp277_confirmatory_raw

    _, paths, beacon_path = gate_b_inputs
    checkpoint = tmp_path / "checkpoint.pt"
    shutil.copyfile(paths["checkpoint"], checkpoint)
    raw_path = tmp_path / "raw-before-analysis.json"
    analysis_path = tmp_path / "analysis-never-written.json"

    def fail_analysis(**_kwargs):
        raise RuntimeError("analysis sentinel failure")

    monkeypatch.setattr(ceremony_module, "build_exp277_confirmatory_analysis", fail_analysis)
    gate_b = _load_script(GATE_B, "exp277_gate_b_raw_before_analysis")
    with pytest.raises(RuntimeError, match="analysis sentinel failure"):
        gate_b.main(
            _gate_b_args(
                paths=paths,
                beacon_path=beacon_path,
                checkpoint_path=checkpoint,
                raw_path=raw_path,
                analysis_path=analysis_path,
            )
        )

    assert raw_path.exists()
    assert not analysis_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    checkpoint_receipt = json.loads(paths["checkpoint_receipt"].read_text(encoding="utf-8"))
    assert validate_exp277_confirmatory_raw(
        raw,
        checkpoint_path=checkpoint,
        checkpoint_receipt=checkpoint_receipt,
    ) == []
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["decision"] == "UNVERIFIED"
