from __future__ import annotations

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_prebranch_observability_v4 import PROBE_KINDS
from nolane_ai.experiments.exp279_prebranch_observability_v4_cross_cell import (
    classify_exp279_prebranch_observability_v4_cross_cell,
)


def _cell(
    *,
    support: bool = True,
    state: bool = False,
    event_mean: bool = False,
    event_residual: bool = False,
) -> dict:
    passes = {
        "STATE_DEEPSETS": state,
        "STATE_EVENT_MEAN": event_mean,
        "STATE_EVENT_RESIDUAL": event_residual,
    }
    return {
        "schema": "NLM-EXP-279-PREBRANCH-OBSERVABILITY-COURT-V4",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "data_boundary": {"evaluation_rng_stream_used": False, "evaluation_targets_used": False},
        "feature_boundary": {"branch_gru_used_for_probe_features": False},
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "probes": {
            kind: {"aggregate": {"support_closed": support}, "economically_routable": passes[kind]}
            for kind in PROBE_KINDS
        },
        "artifact_digest": "a" * 64,
        "court_classification": "fixture",
    }


def test_cross_cell_prefers_robust_state_only_family() -> None:
    result = classify_exp279_prebranch_observability_v4_cross_cell(
        {60: _cell(state=True, event_mean=True, event_residual=True), 120: _cell(state=True, event_mean=True)}
    )
    assert result["decision"] == "ROBUST_STATE_ONLY_PREBRANCH_SIGNAL"
    assert result["robust_probe_families"] == ["STATE_DEEPSETS"]
    assert result["successor_design_authorized"] is True
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False
    assert result["fresh_evaluation_lineage_consumed"] is False


def test_cross_cell_prefers_event_mean_when_state_only_is_not_robust() -> None:
    result = classify_exp279_prebranch_observability_v4_cross_cell(
        {60: _cell(event_mean=True, event_residual=True), 120: _cell(event_mean=True)}
    )
    assert result["decision"] == "ROBUST_EVENT_MEAN_PREBRANCH_SIGNAL"
    assert result["robust_probe_families"] == ["STATE_EVENT_MEAN"]
    assert result["successor_design_authorized"] is True


def test_cross_cell_accepts_event_residual_only_when_same_family_survives_both_cells() -> None:
    robust = classify_exp279_prebranch_observability_v4_cross_cell(
        {60: _cell(event_residual=True), 120: _cell(event_residual=True)}
    )
    mixed = classify_exp279_prebranch_observability_v4_cross_cell(
        {60: _cell(event_mean=True), 120: _cell(event_residual=True)}
    )
    assert robust["decision"] == "ROBUST_EVENT_RESIDUAL_PREBRANCH_SIGNAL"
    assert robust["robust_probe_families"] == ["STATE_EVENT_RESIDUAL"]
    assert robust["successor_design_authorized"] is True
    assert mixed["decision"] == "NO_ROBUST_PREBRANCH_OBSERVABILITY"
    assert mixed["robust_probe_families"] == []
    assert mixed["successor_design_authorized"] is False


def test_cross_cell_no_same_family_win_is_negative() -> None:
    result = classify_exp279_prebranch_observability_v4_cross_cell({60: _cell(), 120: _cell()})
    assert result["decision"] == "NO_ROBUST_PREBRANCH_OBSERVABILITY"
    assert result["successor_design_authorized"] is False


def test_cross_cell_support_failure_is_inconclusive_and_fail_closed() -> None:
    result = classify_exp279_prebranch_observability_v4_cross_cell(
        {60: _cell(support=False, event_residual=True), 120: _cell(event_residual=True)}
    )
    assert result["decision"] == "INCONCLUSIVE_SUPPORT"
    assert result["robust_probe_families"] == []
    assert result["successor_design_authorized"] is False
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False


def test_cross_cell_rejects_missing_decision_cell() -> None:
    with pytest.raises(ValueError, match="train 60 and train 120"):
        classify_exp279_prebranch_observability_v4_cross_cell({60: _cell()})


def test_cross_cell_rejects_evaluation_or_branch_feature_boundary_violation() -> None:
    evaluation_bad = _cell()
    evaluation_bad["data_boundary"]["evaluation_targets_used"] = True
    with pytest.raises(ValueError, match="evaluation boundary"):
        classify_exp279_prebranch_observability_v4_cross_cell({60: evaluation_bad, 120: _cell()})

    feature_bad = _cell()
    feature_bad["feature_boundary"]["branch_gru_used_for_probe_features"] = True
    with pytest.raises(ValueError, match="branch-GRU feature boundary"):
        classify_exp279_prebranch_observability_v4_cross_cell({60: feature_bad, 120: _cell()})
