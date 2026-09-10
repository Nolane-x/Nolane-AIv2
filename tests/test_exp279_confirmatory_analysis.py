from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
PRIMARY_CONTRASTS = [
    "hybrid_vs_propagation_only",
    "hybrid_vs_branch_only",
]


def _fixture(tmp_path):
    from tests.test_exp279_reconstruction_execution import _fixture as build_fixture

    return build_fixture(tmp_path)


def _raw(fixture):
    from nolane_ai.experiments.exp279_confirmatory_executor import (
        execute_exp279_confirmatory_challenge,
    )

    reconstruction = fixture["reconstruction"]
    return execute_exp279_confirmatory_challenge(
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        current_source_tree_digest=reconstruction["source_tree_digest"],
        executor_code_digest="3" * 64,
    )


def _analysis(fixture, raw):
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        build_exp279_confirmatory_analysis,
    )

    return build_exp279_confirmatory_analysis(
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        analysis_code_digest=fixture["seal"]["authorization_snapshot"][
            "analysis_code_digest"
        ],
    )


def test_exp279_blocked_bootstrap_is_deterministic_tests_mesi_boundary_and_has_no_epsilon() -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        bootstrap_exp279_blocked_contrast,
    )

    simple = {stratum: [1.0, 1.0, 1.0, 1.0] for stratum in STRATA}
    hybrid = {stratum: [1.2, 1.2, 1.2, 1.2] for stratum in STRATA}

    result_a = bootstrap_exp279_blocked_contrast(
        simple,
        hybrid,
        seed_material="exp279-blocked-bootstrap-contract",
        samples=500,
        alpha=0.025,
        mesi=0.08,
    )
    result_b = bootstrap_exp279_blocked_contrast(
        simple,
        hybrid,
        seed_material="exp279-blocked-bootstrap-contract",
        samples=500,
        alpha=0.025,
        mesi=0.08,
    )

    assert result_a == result_b
    assert result_a["strata"] == STRATA
    assert result_a["block_weighting"] == "equal_weight_across_predeclared_structure_fit_strata"
    assert result_a["observed_simple_blocked_mean"] == pytest.approx(1.0)
    assert result_a["observed_hybrid_blocked_mean"] == pytest.approx(1.2)
    assert result_a["observed_relative_gain"] == pytest.approx(0.2)
    assert result_a["observed_mesi_excess"] == pytest.approx(0.12)
    assert result_a["mesi_relative_gain"] == pytest.approx(0.08)
    assert result_a["one_sided_p_value"] == pytest.approx(1 / 501)
    assert result_a["one_sided_lower_relative_gain"] == pytest.approx(0.2)
    assert result_a["one_sided_upper_relative_gain"] == pytest.approx(0.2)
    assert result_a["invalid_denominator_resamples"] == 0
    assert result_a["denominator_valid"] is True
    assert result_a["denominator_policy"].startswith("no_epsilon")
    assert result_a["samples"] == 500
    assert result_a["alpha"] == pytest.approx(0.025)
    assert result_a["null_hypothesis"] == "blocked_relative_gain <= 0.08"


def test_exp279_blocked_bootstrap_rejects_nonpositive_scientific_denominator() -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        bootstrap_exp279_blocked_contrast,
    )

    simple = {stratum: [0.0, 0.0] for stratum in STRATA}
    hybrid = {stratum: [1.0, 1.0] for stratum in STRATA}
    result = bootstrap_exp279_blocked_contrast(
        simple,
        hybrid,
        seed_material="exp279-no-epsilon-denominator",
        samples=100,
        alpha=0.025,
        mesi=0.08,
    )
    assert result["denominator_valid"] is False
    assert result["observed_relative_gain"] is None
    assert result["one_sided_p_value"] is None
    assert result["one_sided_lower_relative_gain"] is None
    assert result["one_sided_upper_relative_gain"] is None


def test_exp279_holm_family_uses_frozen_step_down_and_deterministic_tie_order() -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import holm_exp279_family

    both = holm_exp279_family(
        {
            "hybrid_vs_propagation_only": 0.01,
            "hybrid_vs_branch_only": 0.03,
        }
    )
    assert both["ordered_contrasts"] == PRIMARY_CONTRASTS
    assert both["results"]["hybrid_vs_propagation_only"]["threshold"] == pytest.approx(0.025)
    assert both["results"]["hybrid_vs_branch_only"]["threshold"] == pytest.approx(0.05)
    assert both["results"]["hybrid_vs_propagation_only"]["rejected"] is True
    assert both["results"]["hybrid_vs_branch_only"]["rejected"] is True

    sorted_reversal = holm_exp279_family(
        {
            "hybrid_vs_propagation_only": 0.03,
            "hybrid_vs_branch_only": 0.001,
        }
    )
    assert sorted_reversal["ordered_contrasts"] == [
        "hybrid_vs_branch_only",
        "hybrid_vs_propagation_only",
    ]
    assert all(item["rejected"] for item in sorted_reversal["results"].values())

    none = holm_exp279_family(
        {
            "hybrid_vs_propagation_only": 0.03,
            "hybrid_vs_branch_only": 0.04,
        }
    )
    assert not any(item["rejected"] for item in none["results"].values())

    tie = holm_exp279_family(
        {
            "hybrid_vs_branch_only": 0.01,
            "hybrid_vs_propagation_only": 0.01,
        }
    )
    assert tie["ordered_contrasts"] == PRIMARY_CONTRASTS
    assert tie["familywise_alpha"] == pytest.approx(0.05)
    assert tie["step_down_thresholds"] == [0.025, 0.05]


def test_exp279_frozen_decision_rule_exact_boundaries() -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        decide_exp279_confirmatory_outcome,
    )

    assert decide_exp279_confirmatory_outcome(
        selected_holm_rejected=True,
        primary_lower=0.08,
        primary_upper=0.20,
        protected_lower=-0.01,
        protected_upper=0.02,
        denominator_valid=True,
    ) == "PROMOTE_TO_NEXT_STAGE"
    assert decide_exp279_confirmatory_outcome(
        selected_holm_rejected=False,
        primary_lower=0.01,
        primary_upper=0.079,
        protected_lower=0.0,
        protected_upper=0.02,
        denominator_valid=True,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp279_confirmatory_outcome(
        selected_holm_rejected=True,
        primary_lower=0.10,
        primary_upper=0.20,
        protected_lower=-0.03,
        protected_upper=-0.010001,
        denominator_valid=True,
    ) == "KILL_SUBSYSTEM"
    assert decide_exp279_confirmatory_outcome(
        selected_holm_rejected=False,
        primary_lower=0.07,
        primary_upper=0.12,
        protected_lower=-0.005,
        protected_upper=0.01,
        denominator_valid=True,
    ) == "HOLD_UNSTABLE"
    assert decide_exp279_confirmatory_outcome(
        selected_holm_rejected=True,
        primary_lower=None,
        primary_upper=None,
        protected_lower=0.0,
        protected_upper=0.01,
        denominator_valid=False,
    ) == "HOLD_UNSTABLE"


def test_exp279_test_only_analysis_computes_both_contrasts_then_holm_without_promoting(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        validate_exp279_confirmatory_analysis,
    )

    fixture = _fixture(tmp_path)
    raw = _raw(fixture)
    assert raw["status"] == "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"

    analysis = _analysis(fixture, raw)
    assert analysis["schema"] == "NLM-EXP-279-CONFIRMATORY-ANALYSIS-V1"
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

    assert list(analysis["contrasts"]) == PRIMARY_CONTRASTS
    for contrast in PRIMARY_CONTRASTS:
        item = analysis["contrasts"][contrast]
        assert item["effect_type"] == "equal_weight_blocked_ratio_of_means_relative_gain"
        assert item["mesi_relative_gain"] == pytest.approx(0.08)
        assert item["samples"] == 10_000
        assert item["alpha"] == pytest.approx(0.025)
        assert item["strata"] == STRATA
    holm = analysis["holm_family"]
    assert holm["family"] == "PROPAGATION_ROUTING"
    assert set(holm["results"]) == set(PRIMARY_CONTRASTS)
    assert holm["familywise_alpha"] == pytest.approx(0.05)
    assert holm["step_down_thresholds"] == [0.025, 0.05]

    selection = analysis["best_simple_selection"]
    assert selection["selected_arm"] in {"propagation_only", "branch_only"}
    assert selection["selected_contrast"] in PRIMARY_CONTRASTS
    assert selection["selection_applied_after_both_contrasts_and_holm"] is True
    protected = analysis["protected_solution_rate"]
    assert protected["floor_difference"] == pytest.approx(-0.01)
    assert protected["alpha"] == pytest.approx(0.025)
    assert protected["samples"] == 10_000

    assert analysis["wall_energy_per_episode"] == {
        "stage_a_role": "report-only",
        "numeric_measurement": None,
        "fabricated": False,
        "decision_use": False,
        "note": "not measured by EXP-279 confirmatory executor; no numeric value fabricated",
    }
    assert validate_exp279_confirmatory_analysis(
        analysis,
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    ) == []


def test_exp279_analysis_rejects_code_identity_and_rehashed_raw_semantic_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        build_exp279_confirmatory_analysis,
    )
    from nolane_ai.experiments.exp279_confirmatory_executor import _artifact_digest
    from nolane_ai.experiments.exp279_reconstruction_court import _row_digest

    fixture = _fixture(tmp_path)
    raw = _raw(fixture)
    with pytest.raises(ValueError, match="analysis|freeze|code"):
        build_exp279_confirmatory_analysis(
            raw_artifact=raw,
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
            analysis_code_digest="9" * 64,
        )

    forged = deepcopy(raw)
    forged["per_replicate"][0]["hybrid"]["verified_solution_rate"] = 0.123456
    forged["per_replicate"][0]["row_digest"] = _row_digest(
        forged["per_replicate"][0]
    )
    forged["artifact_digest"] = _artifact_digest(forged)
    with pytest.raises(ValueError, match="raw artifact|solution|metric"):
        _analysis(fixture, forged)


def test_exp279_analysis_validator_rejects_rehashed_decision_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_confirmatory_analysis import (
        _analysis_digest,
        validate_exp279_confirmatory_analysis,
    )

    fixture = _fixture(tmp_path)
    raw = _raw(fixture)
    analysis = _analysis(fixture, raw)
    forged = deepcopy(analysis)
    forged["test_only_would_be_decision"] = (
        "KILL_SUBSYSTEM"
        if analysis["test_only_would_be_decision"] != "KILL_SUBSYSTEM"
        else "PROMOTE_TO_NEXT_STAGE"
    )
    forged["analysis_digest"] = _analysis_digest(forged)
    errors = validate_exp279_confirmatory_analysis(
        forged,
        raw_artifact=raw,
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
    )
    assert errors
    assert any("decision" in error.lower() for error in errors)
