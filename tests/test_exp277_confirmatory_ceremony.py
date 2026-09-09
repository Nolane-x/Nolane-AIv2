from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

_support_path = Path(__file__).with_name("test_exp277_reconstruction_court.py")
_support_spec = importlib.util.spec_from_file_location("exp277_ceremony_test_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
court_fixture = _support.court_fixture


def _reconstruction(fixture):
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    return build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )


def _test_beacon():
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt

    return build_test_beacon_receipt(
        source="synthetic-ceremony-beacon",
        beacon_id="ceremony-test-round-277",
        published_at_utc="2026-09-08T10:02:00Z",
        entropy_hex="12" * 32,
        evidence_reference="test-only://exp277-ceremony",
    )


def test_exp277_test_only_ceremony_is_e2_unverified_and_unarmed(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_ceremony import (
        execute_exp277_gate_b_ceremony,
        validate_exp277_gate_b_ceremony,
    )

    fixture = court_fixture
    reconstruction = _reconstruction(fixture)
    ceremony = execute_exp277_gate_b_ceremony(
        seal=fixture["seal"],
        reconstruction_authorization=reconstruction,
        beacon_receipt=_test_beacon(),
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        current_source_tree_digest=reconstruction["source_tree_digest"],
        executor_code_digest="3" * 64,
        test_only=True,
        arm_scientific_lane=False,
    )
    assert ceremony["schema"] == "NLM-EXP-277-CONFIRMATORY-CEREMONY-V1"
    assert ceremony["status"] == "TEST_ONLY_CEREMONY_COMPLETED"
    assert ceremony["scientific_lane_armed"] is False
    assert ceremony["test_only"] is True
    assert ceremony["raw"]["evidence_level"] == "EV-E2"
    assert ceremony["raw"]["decision"] == "UNVERIFIED"
    assert ceremony["analysis"]["evidence_level"] == "EV-E2"
    assert ceremony["analysis"]["decision"] == "UNVERIFIED"
    assert ceremony["analysis"]["scientific_evidence_eligible"] is False
    assert ceremony["analysis"]["decision_rule_executed"] is False
    assert validate_exp277_gate_b_ceremony(
        ceremony,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp277_real_beacon_is_refused_unless_scientific_lane_is_explicitly_armed(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_ceremony import execute_exp277_gate_b_ceremony

    fixture = court_fixture
    reconstruction = _reconstruction(fixture)
    with pytest.raises(RuntimeError, match="scientific lane is not armed"):
        execute_exp277_gate_b_ceremony(
            seal=fixture["seal"],
            reconstruction_authorization=reconstruction,
            beacon_receipt=fixture["beacon"],
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            current_source_tree_digest=reconstruction["source_tree_digest"],
            executor_code_digest="3" * 64,
            test_only=False,
            arm_scientific_lane=False,
        )


def test_exp277_test_only_beacon_cannot_cross_into_scientific_mode(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_ceremony import execute_exp277_gate_b_ceremony

    fixture = court_fixture
    reconstruction = _reconstruction(fixture)
    with pytest.raises(RuntimeError, match="TEST-ONLY"):
        execute_exp277_gate_b_ceremony(
            seal=fixture["seal"],
            reconstruction_authorization=reconstruction,
            beacon_receipt=_test_beacon(),
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            current_source_tree_digest=reconstruction["source_tree_digest"],
            executor_code_digest="3" * 64,
            test_only=False,
            arm_scientific_lane=True,
        )


def test_exp277_ceremony_converts_executor_integrity_failure_to_fail_closed_error(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_ceremony import execute_exp277_gate_b_ceremony

    fixture = court_fixture
    reconstruction = _reconstruction(fixture)
    with pytest.raises(RuntimeError, match="INVALID_RUN"):
        execute_exp277_gate_b_ceremony(
            seal=fixture["seal"],
            reconstruction_authorization=reconstruction,
            beacon_receipt=_test_beacon(),
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            current_source_tree_digest="9" * 64,
            executor_code_digest="3" * 64,
            test_only=True,
            arm_scientific_lane=False,
        )


def test_exp277_gate_b_cli_surface_is_explicit_and_seedless() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_exp277_confirmatory_gate_b.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("run_exp277_confirmatory_gate_b", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
