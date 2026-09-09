from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BUILD_DEV = ROOT / "scripts" / "build_exp279_test_only_development.py"
GATE_A = ROOT / "scripts" / "prepare_exp279_confirmatory_gate_a.py"
GATE_B = ROOT / "scripts" / "run_exp279_confirmatory_gate_b.py"
PROTOCOL_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def _load_script(path: Path, name: str):
    assert path.exists(), f"missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _build_test_only_development(tmp_path: Path) -> tuple[Path, Path]:
    execution = tmp_path / "development.json"
    registry = tmp_path / "registry.json"
    builder = _load_script(BUILD_DEV, "exp279_test_only_dev_builder")
    assert builder.main(
        [
            "--test-only-fixture",
            "--output", str(execution),
            "--registry-output", str(registry),
        ]
    ) == 0
    return execution, registry


def test_exp279_test_only_development_builder_is_canonical_e2_and_non_scientific(
    tmp_path: Path,
) -> None:
    from nolane_ai.experiments.exp279_paired_runner import validate_exp279_paired_development
    from nolane_ai.protocol.identity import source_tree_digest

    execution_path, registry_path = _build_test_only_development(tmp_path)
    execution = _read(execution_path)
    registry = _read(registry_path)
    canonical = PROTOCOL_DIGEST.read_text(encoding="utf-8").strip()

    assert validate_exp279_paired_development(execution) == []
    assert execution["protocol_digest"] == canonical
    assert execution["code_digest"] == source_tree_digest(ROOT)
    assert execution["evidence_level"] == "EV-E2"
    assert execution["decision"] == "UNVERIFIED"
    assert execution["test_only_fixture"] is True
    assert execution["scientific_evidence_eligible"] is False
    assert execution["fixture_purpose"] == "confirmatory_machinery_ci_only"
    assert execution["confirmatory_ready"] is False
    assert execution["confirmatory_data_consumed"] is False
    assert execution["challenge_materialized"] is False
    assert execution["decision_rule_executed"] is False
    assert registry["protocol_digest"] == canonical
    assert (
        registry["experiments"]["EXP-279"]["development_match_status"]
        == "PAIRED_ROUTING_DEV_READY"
    )


def test_exp279_production_gate_a_to_test_only_gate_b_e2e_cannot_promote(
    tmp_path: Path,
) -> None:
    from nolane_ai.experiments.exp279_beacon import build_test_beacon_receipt
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        VALID_SCIENTIFIC_DECISIONS,
        validate_exp279_confirmatory_analysis,
    )
    from nolane_ai.experiments.exp279_confirmatory_authorization import (
        validate_exp279_gate_a_seal,
    )
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        validate_exp279_confirmatory_raw,
    )
    from nolane_ai.experiments.exp279_reconstruction_court import (
        validate_exp279_reconstruction_authorization,
    )

    execution, registry = _build_test_only_development(tmp_path)
    gate_a = _load_script(GATE_A, "exp279_test_only_e2e_gate_a")

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_receipt = tmp_path / "checkpoint-receipt.json"
    prep = tmp_path / "prep.json"
    authorization = tmp_path / "authorization.json"
    seal = tmp_path / "seal.json"
    reconstruction = tmp_path / "reconstruction.json"

    assert gate_a.main(
        [
            "--execution", str(execution),
            "--registry", str(registry),
            "--freeze-commit-sha", "f" * 40,
            "--freeze-commit-timestamp-utc", "2026-09-09T10:00:00Z",
            "--checkpoint-seal-created-at-utc", "2026-09-09T10:00:30Z",
            "--checkpoint-output", str(checkpoint),
            "--checkpoint-receipt-output", str(checkpoint_receipt),
            "--prep-output", str(prep),
            "--authorization-output", str(authorization),
            "--seal-output", str(seal),
            "--reconstruction-output", str(reconstruction),
        ]
    ) == 0

    seal_payload = _read(seal)
    reconstruction_payload = _read(reconstruction)
    checkpoint_receipt_payload = _read(checkpoint_receipt)
    prep_payload = _read(prep)
    assert prep_payload["status"] == "CONFIRMATORY_GATE_A_PREPARED"
    assert prep_payload["confirmatory_ready"] is True
    assert len(prep_payload["confirmatory_lineage"]["reserved_replicate_ids"]) == 32
    assert validate_exp279_gate_a_seal(seal_payload) == []
    assert validate_exp279_reconstruction_authorization(reconstruction_payload) == []
    assert seal_payload["confirmatory_data_consumed"] is False
    assert seal_payload["challenge_materialized"] is False

    beacon = build_test_beacon_receipt(
        source="synthetic-exp279-production-e2e",
        beacon_id="test-only-e2e-round-279",
        published_at_utc="2026-09-09T10:01:00Z",
        entropy_hex="79" * 32,
        evidence_reference="test-only://exp279-production-e2e",
    )
    beacon_path = tmp_path / "beacon.json"
    _write(beacon_path, beacon)

    raw_path = tmp_path / "raw.json"
    analysis_path = tmp_path / "analysis.json"
    gate_b = _load_script(GATE_B, "exp279_test_only_e2e_gate_b")
    assert gate_b.main(
        [
            "--seal", str(seal),
            "--reconstruction", str(reconstruction),
            "--beacon", str(beacon_path),
            "--checkpoint", str(checkpoint),
            "--checkpoint-receipt", str(checkpoint_receipt),
            "--test-only",
            "--raw-output", str(raw_path),
            "--analysis-output", str(analysis_path),
        ]
    ) == 0

    raw = _read(raw_path)
    analysis = _read(analysis_path)
    assert validate_exp279_confirmatory_raw(
        raw,
        checkpoint_path=checkpoint,
        checkpoint_receipt=checkpoint_receipt_payload,
    ) == []
    assert validate_exp279_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=checkpoint,
        checkpoint_receipt=checkpoint_receipt_payload,
    ) == []

    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
    assert raw["test_only"] is True
    assert raw["evidence_level"] == "EV-E2"
    assert raw["decision"] == "UNVERIFIED"
    assert raw["scientific_evidence_eligible"] is False
    assert raw["confirmatory_data_consumed"] is False
    assert raw["synthetic_challenge_data_consumed"] is True
    assert raw["challenge_materialized"] is True
    assert raw["seed_materialization_status"] == "TEST_ONLY_EXECUTED"
    assert raw["decision_rule_executed"] is False

    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["test_only"] is True
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["confirmatory_data_consumed"] is False
    assert analysis["decision_rule_executed"] is False
    assert analysis["test_only_decision_executed"] is True
    assert analysis["test_only_would_be_decision"] in VALID_SCIENTIFIC_DECISIONS
    assert analysis["raw_artifact_digest"] == raw["artifact_digest"]
