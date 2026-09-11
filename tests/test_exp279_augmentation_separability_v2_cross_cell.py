from __future__ import annotations

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_augmentation_separability_v2 import (
    PROBE_KINDS,
    classify_exp279_separability_v2_cross_cell,
)


def _cell(*, support: bool = True, linear: bool = False, mlp: bool = False, deepsets: bool = False) -> dict:
    passes = {
        "LINEAR_MEAN": linear,
        "MLP_MEAN": mlp,
        "DEEPSETS": deepsets,
    }
    return {
        "schema": "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V2",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "data_boundary": {
            "evaluation_rng_stream_used": False,
            "evaluation_targets_used": False,
        },
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "probes": {
            kind: {
                "aggregate": {"support_closed": support},
                "economically_tail_separable": passes[kind],
            }
            for kind in PROBE_KINDS
        },
    }


def test_cross_cell_requires_same_linear_family_at_60_and_120() -> None:
    result = classify_exp279_separability_v2_cross_cell(
        {60: _cell(linear=True), 120: _cell(linear=True)}
    )
    assert result["decision"] == "ROBUST_LINEAR_HEAD_SIGNAL"
    assert result["robust_probe_families"] == ["LINEAR_MEAN"]
    assert result["successor_design_authorized"] is True
    assert result["fresh_evaluation_lineage_consumed"] is False


def test_cross_cell_requires_same_rich_family_not_mixed_cherry_pick() -> None:
    robust = classify_exp279_separability_v2_cross_cell(
        {60: _cell(mlp=True), 120: _cell(mlp=True)}
    )
    mixed = classify_exp279_separability_v2_cross_cell(
        {60: _cell(mlp=True), 120: _cell(deepsets=True)}
    )

    assert robust["decision"] == "ROBUST_RICH_HEAD_SIGNAL"
    assert robust["robust_probe_families"] == ["MLP_MEAN"]
    assert robust["successor_design_authorized"] is True
    assert mixed["decision"] == "NO_ROBUST_HEAD_SIGNAL"
    assert mixed["robust_probe_families"] == []
    assert mixed["successor_design_authorized"] is False


def test_cross_cell_support_failure_is_inconclusive_and_fail_closed() -> None:
    result = classify_exp279_separability_v2_cross_cell(
        {60: _cell(support=False, linear=True), 120: _cell(linear=True)}
    )
    assert result["decision"] == "INCONCLUSIVE_SUPPORT"
    assert result["successor_design_authorized"] is False
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False


def test_cross_cell_no_robust_family_keeps_fresh_lineage_locked() -> None:
    result = classify_exp279_separability_v2_cross_cell(
        {60: _cell(), 120: _cell()}
    )
    assert result["decision"] == "NO_ROBUST_HEAD_SIGNAL"
    assert result["successor_design_authorized"] is False
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False
    assert result["fresh_evaluation_lineage_consumed"] is False
    assert result["confirmatory_data_consumed"] is False


def test_cross_cell_rejects_missing_decision_cell() -> None:
    with pytest.raises(ValueError, match="train 60 and train 120"):
        classify_exp279_separability_v2_cross_cell({60: _cell()})
