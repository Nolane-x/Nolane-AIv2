from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

_support_path = Path(__file__).with_name("test_exp277_reconstruction_court.py")
_support_spec = importlib.util.spec_from_file_location("exp277_analysis_test_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)
court_fixture = _support.court_fixture


def _ensure_reconstruction(fixture) -> None:
    if "reconstruction" in fixture:
        return
    from nolane_ai.experiments.exp277_reconstruction_court import build_exp277_reconstruction_authorization

    fixture["reconstruction"] = build_exp277_reconstruction_authorization(
        seal=fixture["seal"],
        reconstruction_code_digest="4" * 64,
    )


def _raw(fixture, *, test_only: bool = False):
    from nolane_ai.experiments.exp277_beacon import build_test_beacon_receipt
    from nolane_ai.experiments.exp277_confirmatory_executor import execute_exp277_confirmatory_challenge

    _ensure_reconstruction(fixture)
    beacon = fixture["beacon"]
    if test_only:
        beacon = build_test_beacon_receipt(
            source="synthetic-analysis-test-beacon",
            beacon_id="analysis-test-round-277",
            published_at_utc="2026-09-08T10:02:00Z",
            entropy_hex="ef" * 32,
            evidence_reference="test-only://exp277-analysis",
        )
    return execute_exp277_confirmatory_challenge(
        reconstruction_authorization=fixture["reconstruction"],
        seal=fixture["seal"],
        beacon_receipt=beacon,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        current_source_tree_digest=fixture["reconstruction"]["source_tree_digest"],
        executor_code_digest="3" * 64,
    )


def _analysis(fixture, raw):
    from nolane_ai.experiments.exp277_confirmatory_analysis import build_exp277_confirmatory_analysis

    analysis_code_digest = fixture["seal"]["authorization_snapshot"]["analysis_code_digest"]
    return build_exp277_confirmatory_analysis(
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        analysis_code_digest=analysis_code_digest,
    )


def test_exp277_ratio_of_means_bootstrap_is_deterministic_and_has_no_epsilon() -> None:
    from nolane_ai.experiments.exp277_confirmatory_analysis import bootstrap_exp277_ratio_of_means

    result_a = bootstrap_exp277_ratio_of_means(
        [1.0, 1.0, 1.0, 1.0],
        [1.2, 1.2, 1.2, 1.2],
        seed_material="exp277-bootstrap-contract",
        samples=500,
        alpha=0.025,
    )
    result_b = bootstrap_exp277_ratio_of_means(
        [1.0, 1.0, 1.0, 1.0],
        [1.2, 1.2, 1.2, 1.2],
        seed_material="exp277-bootstrap-contract",
        samples=500,
        alpha=0.025,
    )
    assert result_a == result_b
    assert result_a["observed_baseline_mean"] == pytest.approx(1.0)
    assert result_a["observed_candidate_mean"] == pytest.approx(1.2)
    assert result_a["observed_relative_gain"] == pytest.approx(0.2)
    assert result_a["one_sided_lower"] == pytest.approx(0.2)
    assert result_a["one_sided_upper"] == pytest.approx(0.2)
    assert result_a["invalid_denominator_resamples"] == 0
    assert result_a["denominator_policy"] == "no_epsilon"


def test_exp277_frozen_decision_rule_exact_boundaries_and_denominator_hold() -> None:
    from nolane_ai.experiments.exp277_confirmatory_analysis import decide_exp277_confirmatory_outcome

    assert decide_exp277_confirmatory_outcome(
        primary_lower=0.10,
        primary_upper=0.20,
        solution_lower=-0.005,
        solution_upper=0.01,
        denominator_valid=True,
    ) == "PROMOTE_TO_NEXT_STAGE"
    assert decide_exp277_confirmatory_outcome(
        primary_lower=0.01,
        primary_upper=0.099,
        solution_lower=-0.001,
        solution_upper=0.01,
        denominator_valid=True,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp277_confirmatory_outcome(
        primary_lower=0.20,
        primary_upper=0.30,
        solution_lower=-0.02,
        solution_upper=-0.006,
        denominator_valid=True,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp277_confirmatory_outcome(
        primary_lower=0.08,
        primary_upper=0.12,
        solution_lower=-0.01,
        solution_upper=0.01,
        denominator_valid=True,
    ) == "HOLD_UNSTABLE"
    assert decide_exp277_confirmatory_outcome(
        primary_lower=0.20,
        primary_upper=0.30,
        solution_lower=0.0,
        solution_upper=0.01,
        denominator_valid=False,
    ) == "HOLD_UNSTABLE"
    assert decide_exp277_confirmatory_outcome(
        primary_lower=None,
        primary_upper=0.09,
        solution_lower=0.0,
        solution_upper=0.01,
        denominator_valid=False,
    ) == "KILL_SUBSYSTEM"


def test_exp277_scientific_analysis_executes_only_frozen_rule_and_reports_cost_boundary(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_analysis import validate_exp277_confirmatory_analysis

    fixture = court_fixture
    raw = _raw(fixture)
    assert raw["status"] == "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
    analysis = _analysis(fixture, raw)
    assert analysis["schema"] == "NLM-EXP-277-CONFIRMATORY-ANALYSIS-V1"
    assert analysis["status"] == "CONFIRMATORY_CHALLENGE_ANALYZED"
    assert analysis["evidence_level"] == "EV-E3"
    assert analysis["decision"] in {"PROMOTE_TO_NEXT_STAGE", "HOLD_UNSTABLE", "KILL_SUBSYSTEM"}
    assert analysis["scientific_evidence_eligible"] is True
    assert analysis["confirmatory_data_consumed"] is True
    assert analysis["decision_rule_executed"] is True
    assert analysis["test_only"] is False
    primary = analysis["primary_endpoint"]
    assert primary["metric"] == "verified_utility_per_accounted_flop"
    assert primary["effect_type"] == "ratio_of_means_relative_gain"
    assert primary["mesi_relative_gain"] == 0.10
    assert primary["alpha"] == 0.025
    assert primary["samples"] == 10_000
    solution = analysis["protected_solution_rate"]
    assert solution["floor_difference"] == -0.005
    assert solution["alpha"] == 0.025
    wall = analysis["wall_energy_per_episode"]
    assert wall == {
        "stage_a_role": "report-only",
        "numeric_measurement": None,
        "fabricated": False,
        "decision_use": False,
        "note": "not measured by EXP-277 confirmatory executor; no numeric value fabricated",
    }
    assert validate_exp277_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp277_test_only_analysis_cannot_create_ev_e3_or_scientific_decision(court_fixture) -> None:
    fixture = court_fixture
    raw = _raw(fixture, test_only=True)
    analysis = _analysis(fixture, raw)
    assert analysis["status"] == "TEST_ONLY_CHALLENGE_ANALYZED"
    assert analysis["evidence_level"] == "EV-E2"
    assert analysis["decision"] == "UNVERIFIED"
    assert analysis["scientific_evidence_eligible"] is False
    assert analysis["confirmatory_data_consumed"] is False
    assert analysis["synthetic_challenge_data_consumed"] is True
    assert analysis["decision_rule_executed"] is False
    assert analysis["test_only_decision_executed"] is True
    assert analysis["test_only_would_be_decision"] in {
        "PROMOTE_TO_NEXT_STAGE",
        "HOLD_UNSTABLE",
        "KILL_SUBSYSTEM",
    }


def test_exp277_analysis_rejects_code_identity_drift_before_decision(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_analysis import build_exp277_confirmatory_analysis

    fixture = court_fixture
    raw = _raw(fixture)
    with pytest.raises(ValueError, match="pre-beacon freeze"):
        build_exp277_confirmatory_analysis(
            raw_artifact=raw,
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            analysis_code_digest="9" * 64,
        )


def test_exp277_analysis_rejects_rehashed_raw_semantic_tamper(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_executor import _artifact_digest
    from nolane_ai.experiments.exp277_reconstruction_court import _row_digest

    fixture = court_fixture
    raw = _raw(fixture)
    forged = deepcopy(raw)
    forged["per_replicate"][0]["arcs_branch"]["verified_solution_rate"] = 0.123
    forged["per_replicate"][0]["row_digest"] = _row_digest(forged["per_replicate"][0])
    forged["artifact_digest"] = _artifact_digest(forged)
    with pytest.raises(ValueError, match="raw artifact"):
        _analysis(fixture, forged)


def test_exp277_analysis_validator_rejects_rehashed_decision_tamper(court_fixture) -> None:
    from nolane_ai.experiments.exp277_confirmatory_analysis import (
        _analysis_digest,
        validate_exp277_confirmatory_analysis,
    )

    fixture = court_fixture
    raw = _raw(fixture)
    analysis = _analysis(fixture, raw)
    forged = deepcopy(analysis)
    forged["decision"] = (
        "KILL_SUBSYSTEM" if analysis["decision"] != "KILL_SUBSYSTEM" else "PROMOTE_TO_NEXT_STAGE"
    )
    forged["analysis_digest"] = _analysis_digest(forged)
    errors = validate_exp277_confirmatory_analysis(
        forged,
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert errors
    assert any("decision" in error.lower() for error in errors)
