from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp279_counterfactual_delta_distill_v6_cross_cell import (
    classify_exp279_counterfactual_delta_distill_v6_cross_cell,
)


def _cell(
    train_replicates: int,
    *,
    fit_support: bool = True,
    heldout_support: bool = True,
    cdd_improved: bool = False,
    control_improved: bool = False,
) -> dict:
    return {
        "schema": "NLM-EXP-279-COUNTERFACTUAL-DELTA-DISTILL-COURT-V6",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "experiment_id": "EXP-279",
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440",
        "code_digest": "same-code-digest",
        "root_seed": "20260912-exp279-counterfactual-delta-distill-v6-dev",
        "canonical_model_frozen_for_cdd": True,
        "data_boundary": {
            "training_rng_stream": "augmentation",
            "probe_rng_stream": "augmentation",
            "probe_root_seed": "20260912-exp279-counterfactual-delta-distill-v6-dev::independent-augmentation-cdd-v6",
            "probe_root_independent_from_training_root": True,
            "evaluation_rng_stream_used": False,
            "evaluation_targets_used": False,
            "heldout_branch_hidden_used_for_features": False,
            "confirmatory_examples_used": False,
            "external_examples_used": False,
        },
        "canonical_training": {
            "replicates": train_replicates,
            "rng_stream": "augmentation",
            "canonical_parameters_frozen_before_cdd": True,
        },
        "teacher_rule": {
            "teacher_uses_fit_partition_only": True,
            "heldout_teacher_delta_materialized_for_features": False,
        },
        "probe_config": {
            "primary_family": "CDD_DELTA_LINEAR",
            "descriptive_control_family": "RAW_CHEAP_LINEAR",
            "control_can_authorize_successor": False,
            "raw_scores_compared_across_folds": False,
            "raw_scores_exported": False,
        },
        "cdd_cost_rule": {
            "cdd_inference_flops_charged_to_primary_utility": True,
            "cdd_cost_charged_on_stop_and_branch_paths": True,
            "false_positive_harms_counted_directly": True,
        },
        "probes": {
            "CDD_DELTA_LINEAR": {
                "folds": [
                    {
                        "fit_supported": fit_support,
                        "distiller_frozen_before_selector": True,
                    }
                    for _ in range(3)
                ],
                "aggregate": {
                    "support_closed": heldout_support,
                    "direct_utility_improved": cdd_improved,
                },
                "authorizes_successor": cdd_improved,
                "raw_scores_compared_across_folds": False,
            },
            "RAW_CHEAP_LINEAR": {
                "folds": [{"fit_supported": fit_support} for _ in range(3)],
                "aggregate": {
                    "support_closed": heldout_support,
                    "direct_utility_improved": control_improved,
                },
                "authorizes_successor": False,
                "raw_scores_compared_across_folds": False,
            },
        },
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "artifact_digest": f"cell-{train_replicates}",
    }


def test_cross_cell_requires_cdd_direct_utility_positive_at_60_and_120() -> None:
    result = classify_exp279_counterfactual_delta_distill_v6_cross_cell(
        {60: _cell(60, cdd_improved=True), 120: _cell(120, cdd_improved=True)}
    )
    assert result["decision"] == "ROBUST_CDD_BRANCH_COMPLEMENTARITY"
    assert result["successor_design_authorized"] is True
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False
    assert result["fresh_evaluation_lineage_consumed"] is False


def test_cross_cell_control_positivity_cannot_authorize_successor() -> None:
    result = classify_exp279_counterfactual_delta_distill_v6_cross_cell(
        {
            60: _cell(60, cdd_improved=False, control_improved=True),
            120: _cell(120, cdd_improved=False, control_improved=True),
        }
    )
    assert result["decision"] == "NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY"
    assert result["successor_design_authorized"] is False
    assert result["control_can_authorize_successor"] is False


def test_cross_cell_mixed_cdd_cells_are_not_robust() -> None:
    result = classify_exp279_counterfactual_delta_distill_v6_cross_cell(
        {60: _cell(60, cdd_improved=True), 120: _cell(120, cdd_improved=False)}
    )
    assert result["decision"] == "NO_ROBUST_CDD_BRANCH_COMPLEMENTARITY"
    assert result["successor_design_authorized"] is False


@pytest.mark.parametrize(
    ("fit_support", "heldout_support"),
    [(False, True), (True, False)],
)
def test_cross_cell_missing_fit_or_heldout_support_is_inconclusive(
    fit_support: bool,
    heldout_support: bool,
) -> None:
    result = classify_exp279_counterfactual_delta_distill_v6_cross_cell(
        {
            60: _cell(
                60,
                fit_support=fit_support,
                heldout_support=heldout_support,
                cdd_improved=True,
            ),
            120: _cell(120, cdd_improved=True),
        }
    )
    assert result["decision"] == "INCONCLUSIVE_SUPPORT"
    assert result["successor_design_authorized"] is False
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False


def test_cross_cell_rejects_missing_or_mislabeled_decision_cells() -> None:
    with pytest.raises(ValueError, match="train 60 and train 120"):
        classify_exp279_counterfactual_delta_distill_v6_cross_cell({60: _cell(60)})

    with pytest.raises(ValueError, match="replicate mismatch"):
        classify_exp279_counterfactual_delta_distill_v6_cross_cell(
            {60: _cell(120), 120: _cell(120)}
        )


def test_cross_cell_rejects_schema_evidence_and_provenance_mismatch() -> None:
    for field, value, match in (
        ("schema", "wrong", "schema mismatch"),
        ("evidence_level", "EV-E3", "evidence boundary mismatch"),
        ("decision", "CONFIRMED", "evidence boundary mismatch"),
        ("scientific_evidence_eligible", True, "evidence boundary mismatch"),
        ("protocol_digest", "other", "provenance mismatch"),
        ("code_digest", "other", "provenance mismatch"),
        ("root_seed", "other", "provenance mismatch"),
    ):
        cell60 = _cell(60)
        cell120 = _cell(120)
        cell120[field] = value
        with pytest.raises(ValueError, match=match):
            classify_exp279_counterfactual_delta_distill_v6_cross_cell(
                {60: cell60, 120: cell120}
            )


def test_cross_cell_rejects_evaluation_confirmatory_or_fresh_lineage_violation() -> None:
    mutations = [
        (("data_boundary", "evaluation_rng_stream_used"), True),
        (("data_boundary", "evaluation_targets_used"), True),
        (("data_boundary", "heldout_branch_hidden_used_for_features"), True),
        (("data_boundary", "confirmatory_examples_used"), True),
        (("teacher_rule", "teacher_uses_fit_partition_only"), False),
        (("teacher_rule", "heldout_teacher_delta_materialized_for_features"), True),
        (("cdd_cost_rule", "cdd_inference_flops_charged_to_primary_utility"), False),
        (("probe_config", "raw_scores_exported"), True),
        (("probe_config", "raw_scores_compared_across_folds"), True),
        (("fresh_evaluation_lineage_may_be_reserved",), True),
        (("fresh_evaluation_lineage_consumed",), True),
        (("confirmatory_data_consumed",), True),
        (("challenge_materialized",), True),
        (("promotion_claimed",), True),
    ]
    for path, value in mutations:
        bad = copy.deepcopy(_cell(60))
        cursor = bad
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match="boundary violated"):
            classify_exp279_counterfactual_delta_distill_v6_cross_cell(
                {60: bad, 120: _cell(120)}
            )


def test_cross_cell_receipt_is_development_only_even_when_robust() -> None:
    result = classify_exp279_counterfactual_delta_distill_v6_cross_cell(
        {60: _cell(60, cdd_improved=True), 120: _cell(120, cdd_improved=True)}
    )
    assert result["schema"] == "NLM-EXP-279-COUNTERFACTUAL-DELTA-DISTILL-CROSS-CELL-V6"
    assert result["evidence_level"] == "EV-E2"
    assert result["scientific_evidence_eligible"] is False
    assert result["decision_train_replicates"] == [60, 120]
    assert result["control_can_authorize_successor"] is False
    assert result["fresh_evaluation_lineage_may_be_reserved"] is False
    assert result["fresh_evaluation_lineage_consumed"] is False
    assert result["confirmatory_data_consumed"] is False
    assert result["challenge_materialized"] is False
    assert result["promotion_claimed"] is False
