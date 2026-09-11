from __future__ import annotations

import copy

import pytest

from nolane_ai.experiments.exp279_cost_accounted_branch_preview_v5_cross_cell import (
    CROSS_CELL_SCHEMA,
    classify_exp279_cost_accounted_branch_preview_v5_cross_cell,
)


FAMILIES = (
    "PREFIX1_STATE_LINEAR",
    "PREFIX2_STATE_LINEAR",
    "PREFIX3_STATE_LINEAR",
)


def _cell(*, train_steps: int, positives: tuple[str, ...] = FAMILIES) -> dict:
    probes = {
        family: {
            "aggregate": {
                "support_closed": True,
                "direct_utility_improved": family in positives,
            }
        }
        for family in FAMILIES
    }
    return {
        "schema": "NLM-EXP-279-COST-ACCOUNTED-BRANCH-PREVIEW-COURT-V5",
        "train_steps": train_steps,
        "evidence_state": "EV-E2 / UNVERIFIED",
        "scientific_evidence_eligible": False,
        "data_boundary": {
            "evaluation_rng_stream_used": False,
            "evaluation_targets_used": False,
        },
        "preview_cost_rule": {
            "preview_flops_charged_to_primary_utility": True,
            "selector_flops_charged_to_primary_utility": True,
        },
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "probes": probes,
    }


def _classify(train60: dict, train120: dict) -> dict:
    return classify_exp279_cost_accounted_branch_preview_v5_cross_cell([train60, train120])


@pytest.mark.parametrize(
    ("family", "decision"),
    [
        ("PREFIX1_STATE_LINEAR", "ROBUST_PREFIX1_BRANCH_PREVIEW"),
        ("PREFIX2_STATE_LINEAR", "ROBUST_PREFIX2_BRANCH_PREVIEW"),
        ("PREFIX3_STATE_LINEAR", "ROBUST_PREFIX3_BRANCH_PREVIEW"),
    ],
)
def test_same_family_positive_in_both_decision_cells_survives(family: str, decision: str) -> None:
    out = _classify(
        _cell(train_steps=60, positives=(family,)),
        _cell(train_steps=120, positives=(family,)),
    )
    assert out["schema"] == CROSS_CELL_SCHEMA
    assert out["decision"] == decision
    assert out["robust_preview_families"] == [family]
    assert out["successor_design_authorized"] is True
    assert out["fresh_evaluation_lineage_may_be_reserved"] is False
    assert out["fresh_evaluation_lineage_consumed"] is False


def test_cheapest_first_hierarchy_is_deterministic() -> None:
    out = _classify(_cell(train_steps=60), _cell(train_steps=120))
    assert out["decision"] == "ROBUST_PREFIX1_BRANCH_PREVIEW"
    assert out["robust_preview_families"] == list(FAMILIES)


def test_mixed_family_positives_do_not_survive_cross_cell() -> None:
    out = _classify(
        _cell(train_steps=60, positives=("PREFIX1_STATE_LINEAR",)),
        _cell(train_steps=120, positives=("PREFIX2_STATE_LINEAR",)),
    )
    assert out["decision"] == "NO_ROBUST_COST_ACCOUNTED_BRANCH_PREVIEW"
    assert out["robust_preview_families"] == []
    assert out["successor_design_authorized"] is False


def test_open_support_is_inconclusive_not_negative() -> None:
    train60 = _cell(train_steps=60, positives=("PREFIX1_STATE_LINEAR",))
    train60["probes"]["PREFIX1_STATE_LINEAR"]["aggregate"]["support_closed"] = False
    out = _classify(train60, _cell(train_steps=120, positives=("PREFIX1_STATE_LINEAR",)))
    assert out["decision"] == "INCONCLUSIVE_SUPPORT"
    assert out["successor_design_authorized"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cell: cell["data_boundary"].__setitem__("evaluation_targets_used", True),
        lambda cell: cell["data_boundary"].__setitem__("evaluation_rng_stream_used", True),
        lambda cell: cell.__setitem__("fresh_evaluation_lineage_may_be_reserved", True),
        lambda cell: cell.__setitem__("fresh_evaluation_lineage_consumed", True),
        lambda cell: cell.__setitem__("confirmatory_data_consumed", True),
        lambda cell: cell.__setitem__("challenge_materialized", True),
        lambda cell: cell.__setitem__("promotion_claimed", True),
        lambda cell: cell["preview_cost_rule"].__setitem__("preview_flops_charged_to_primary_utility", False),
        lambda cell: cell["preview_cost_rule"].__setitem__("selector_flops_charged_to_primary_utility", False),
    ],
)
def test_boundary_violations_fail_closed(mutate) -> None:
    train60 = _cell(train_steps=60, positives=("PREFIX1_STATE_LINEAR",))
    mutate(train60)
    with pytest.raises(ValueError):
        _classify(train60, _cell(train_steps=120, positives=("PREFIX1_STATE_LINEAR",)))


def test_requires_exactly_train60_and_train120() -> None:
    with pytest.raises(ValueError):
        classify_exp279_cost_accounted_branch_preview_v5_cross_cell([_cell(train_steps=60)])
    with pytest.raises(ValueError):
        _classify(_cell(train_steps=15), _cell(train_steps=120))


def test_classifier_does_not_mutate_inputs() -> None:
    cells = [_cell(train_steps=60), _cell(train_steps=120)]
    before = copy.deepcopy(cells)
    _ = classify_exp279_cost_accounted_branch_preview_v5_cross_cell(cells)
    assert cells == before
