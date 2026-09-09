from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GATE_B = ROOT / "scripts" / "run_exp279_confirmatory_gate_b.py"
PROTOCOL = ROOT / "protocols" / "stage_a_v1.json"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _load_gate_b(name: str):
    assert GATE_B.exists(), f"missing script: {GATE_B}"
    spec = importlib.util.spec_from_file_location(name, GATE_B)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    from tests.test_exp279_reconstruction_execution import _fixture

    fixture = _fixture(tmp_path)
    paths = {
        "seal": tmp_path / "seal.json",
        "reconstruction": tmp_path / "reconstruction.json",
        "beacon": tmp_path / "beacon.json",
        "checkpoint_receipt": tmp_path / "checkpoint-receipt.json",
    }
    _write_json(paths["seal"], fixture["seal"])
    _write_json(paths["reconstruction"], fixture["reconstruction"])
    _write_json(paths["beacon"], fixture["beacon"])
    _write_json(paths["checkpoint_receipt"], fixture["checkpoint_receipt"])
    return fixture, paths


def _bind_synthetic_fixture_identity(
    module,
    fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_tree_digest: str | None = None,
) -> None:
    # Unit fixtures intentionally use synthetic protocol/source identities. The
    # CLI's real canonical protocol court is tested separately below; these
    # persistence tests bind the synthetic fixture lineage so execution can
    # reach the raw-before-analysis boundary they are designed to exercise.
    monkeypatch.setattr(
        module,
        "_verified_protocol",
        lambda _protocol, _digest: fixture["reconstruction"]["protocol_digest"],
    )
    effective_source = (
        fixture["reconstruction"]["source_tree_digest"]
        if source_tree_digest is None
        else source_tree_digest
    )
    monkeypatch.setattr(module, "source_tree_digest", lambda _root: effective_source)


def _args(
    *,
    fixture: dict,
    paths: dict[str, Path],
    raw_output: Path,
    analysis_output: Path,
) -> list[str]:
    return [
        "--protocol", str(PROTOCOL),
        "--protocol-digest-file", str(PROTOCOL_DIGEST),
        "--seal", str(paths["seal"]),
        "--reconstruction", str(paths["reconstruction"]),
        "--beacon", str(paths["beacon"]),
        "--checkpoint", str(fixture["checkpoint_path"]),
        "--checkpoint-receipt", str(paths["checkpoint_receipt"]),
        "--test-only",
        "--raw-output", str(raw_output),
        "--analysis-output", str(analysis_output),
    ]


def test_exp279_gate_b_cli_surface_is_explicit_mutually_exclusive_seedless_and_canonical() -> None:
    module = _load_gate_b("exp279_gate_b_cli_surface")

    expected_protocol_digest = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()
    assert module._verified_protocol(PROTOCOL, PROTOCOL_DIGEST) == expected_protocol_digest

    with pytest.raises(SystemExit, match="unrecognized arguments"):
        module.parse_args(["--seed", "123"])

    args = module.parse_args(
        [
            "--seal", "seal.json",
            "--reconstruction", "reconstruction.json",
            "--beacon", "beacon.json",
            "--checkpoint", "checkpoint.pt",
            "--checkpoint-receipt", "checkpoint.json",
            "--test-only",
            "--raw-output", "raw.json",
            "--analysis-output", "analysis.json",
        ]
    )
    assert args.test_only is True
    assert args.arm_scientific_lane is False

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--seal", "seal.json",
                "--reconstruction", "reconstruction.json",
                "--beacon", "beacon.json",
                "--checkpoint", "checkpoint.pt",
                "--checkpoint-receipt", "checkpoint.json",
                "--test-only",
                "--arm-scientific-lane",
                "--raw-output", "raw.json",
                "--analysis-output", "analysis.json",
            ]
        )


def test_exp279_gate_b_cli_persists_valid_test_only_raw_then_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        validate_exp279_confirmatory_analysis,
    )
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        validate_exp279_confirmatory_raw,
    )

    fixture, paths = _inputs(tmp_path)
    raw_path = tmp_path / "raw.json"
    analysis_path = tmp_path / "analysis.json"
    module = _load_gate_b("exp279_gate_b_cli_valid")
    _bind_synthetic_fixture_identity(module, fixture, monkeypatch)

    assert module.main(
        _args(
            fixture=fixture,
            paths=paths,
            raw_output=raw_path,
            analysis_output=analysis_path,
        )
    ) == 0

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert validate_exp279_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []
    assert validate_exp279_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["decision_rule_executed"] is False
    assert analysis["raw_artifact_digest"] == raw["artifact_digest"]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_exp279_gate_b_cli_persists_invalid_raw_and_never_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        validate_exp279_confirmatory_raw,
    )

    fixture, paths = _inputs(tmp_path)
    raw_path = tmp_path / "invalid-raw.json"
    analysis_path = tmp_path / "invalid-analysis.json"
    module = _load_gate_b("exp279_gate_b_cli_invalid")
    _bind_synthetic_fixture_identity(
        module,
        fixture,
        monkeypatch,
        source_tree_digest="9" * 64,
    )

    with pytest.raises(RuntimeError, match="INVALID_RUN"):
        module.main(
            _args(
                fixture=fixture,
                paths=paths,
                raw_output=raw_path,
                analysis_output=analysis_path,
            )
        )

    assert raw_path.exists()
    assert not analysis_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert validate_exp279_confirmatory_raw(
        raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []
    assert raw["status"] == "INVALID_RUN"
    assert raw["decision"] == "INVALID_RUN"
    assert raw["scientific_evidence_eligible"] is False
    assert raw["challenge_materialized"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["per_replicate"] == []
    assert raw["integrity_errors"]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_exp279_gate_b_cli_keeps_raw_when_analysis_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nolane_ai.experiments.exp279_confirmatory_ceremony as ceremony_module

    fixture, paths = _inputs(tmp_path)
    raw_path = tmp_path / "raw-before-analysis.json"
    analysis_path = tmp_path / "analysis-never-written.json"
    module = _load_gate_b("exp279_gate_b_cli_analysis_failure")
    _bind_synthetic_fixture_identity(module, fixture, monkeypatch)

    def fail_analysis(**_kwargs):
        raise RuntimeError("analysis sentinel failure")

    monkeypatch.setattr(
        ceremony_module,
        "build_exp279_confirmatory_analysis",
        fail_analysis,
    )
    with pytest.raises(RuntimeError, match="analysis sentinel failure"):
        module.main(
            _args(
                fixture=fixture,
                paths=paths,
                raw_output=raw_path,
                analysis_output=analysis_path,
            )
        )

    assert raw_path.exists()
    assert not analysis_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["decision"] == "UNVERIFIED"
    assert list(tmp_path.glob(".*.tmp")) == []
