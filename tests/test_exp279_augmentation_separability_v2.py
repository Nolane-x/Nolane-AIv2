from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_augmentation_separability_v2 import (
    PROBE_KINDS,
    _aggregate_fold_ranking_metrics,
    _assign_probe_fold,
    _economic_break_even_probability,
    _fold_ranking_metrics,
    run_exp279_augmentation_separability_v2_court,
    validate_exp279_augmentation_separability_v2_court,
)


def test_fold_assignment_balances_all_three_strata_per_fold_block() -> None:
    assert [_assign_probe_fold(replicate, folds=3) for replicate in range(9)] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2,
    ]


def test_fold_local_aggregation_is_invariant_to_independent_positive_affine_score_changes() -> None:
    labels = [
        torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        torch.tensor([1, 0, 0, 1, 0], dtype=torch.long),
        torch.tensor([0, 0, 1, 0, 1], dtype=torch.long),
    ]
    scores = [
        torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float64),
        torch.tensor([0.7, 0.4, 0.2, 0.8, 0.1], dtype=torch.float64),
        torch.tensor([0.1, 0.2, 0.8, 0.3, 0.9], dtype=torch.float64),
    ]
    transformed = [
        scores[0] * 7.0 + 100.0,
        scores[1] * 0.2 - 37.0,
        scores[2] * 3.0 + 5.0,
    ]

    original_rows = [_fold_ranking_metrics(score, label) for score, label in zip(scores, labels)]
    shifted_rows = [
        _fold_ranking_metrics(score, label)
        for score, label in zip(transformed, labels)
    ]
    original = _aggregate_fold_ranking_metrics(original_rows)
    shifted = _aggregate_fold_ranking_metrics(shifted_rows)

    assert original["cross_fitted_roc_auc"] == pytest.approx(
        shifted["cross_fitted_roc_auc"], abs=1e-12
    )
    assert original["cross_fitted_tail_precision"] == pytest.approx(
        shifted["cross_fitted_tail_precision"], abs=1e-12
    )
    assert original["tail_selected_rescues"] == shifted["tail_selected_rescues"]
    assert original["tail_selected_episodes"] == shifted["tail_selected_episodes"]


def test_fold_local_aggregate_uses_only_within_fold_pairs_and_fold_local_k() -> None:
    rows = [
        _fold_ranking_metrics(
            torch.tensor([0.9, 0.8, 0.1], dtype=torch.float64),
            torch.tensor([1, 0, 0], dtype=torch.long),
        ),
        _fold_ranking_metrics(
            torch.tensor([0.1, 0.8, 0.9, 0.2], dtype=torch.float64),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ),
    ]
    aggregate = _aggregate_fold_ranking_metrics(rows)

    expected_auc = (1.0 * 2 + 0.5 * 4) / 6
    assert aggregate["cross_fitted_roc_auc"] == pytest.approx(expected_auc, abs=1e-12)
    assert aggregate["tail_selected_episodes"] == 3
    assert aggregate["tail_selected_rescues"] == 2
    assert aggregate["cross_fitted_tail_precision"] == pytest.approx(2 / 3, abs=1e-12)
    assert aggregate["raw_scores_compared_across_folds"] is False


def test_missing_class_support_fails_closed() -> None:
    rows = [
        _fold_ranking_metrics(
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
            torch.tensor([0, 0, 0], dtype=torch.long),
        ),
        _fold_ranking_metrics(
            torch.tensor([0.9, 0.1], dtype=torch.float64),
            torch.tensor([1, 0], dtype=torch.long),
        ),
    ]
    aggregate = _aggregate_fold_ranking_metrics(rows)

    assert aggregate["support_closed"] is False
    assert aggregate["cross_fitted_roc_auc"] is None
    assert aggregate["cross_fitted_tail_precision"] is None


def test_break_even_is_derived_from_sealed_compute_ledger() -> None:
    value = _economic_break_even_probability(
        stop_accounted_flops_per_episode=110_046.0,
        branch_accounted_flops_per_episode=224_766.0,
    )
    incremental = (224_766.0 - 110_046.0) / 224_766.0
    assert value == pytest.approx(incremental / (1.0 + incremental), abs=1e-15)


def test_probe_family_and_no_cross_fold_score_alignment_are_frozen() -> None:
    assert PROBE_KINDS == ("LINEAR_MEAN", "MLP_MEAN", "DEEPSETS")


def test_tiny_v2_court_is_augmentation_only_and_never_reserves_eval_on_invalid_support() -> None:
    payload = run_exp279_augmentation_separability_v2_court(
        root_seed="exp279-separability-v2-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        probe_replicates=9,
        probe_folds=3,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        probe_steps=1,
        probe_lr=1e-2,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    assert payload["schema"] == "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V2"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["data_boundary"]["training_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["probe_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["evaluation_rng_stream_used"] is False
    assert payload["data_boundary"]["evaluation_targets_used"] is False
    assert payload["data_boundary"]["probe_root_seed"].endswith(
        "::independent-augmentation-probe-v2"
    )
    assert payload["probe_config"]["raw_scores_compared_across_folds"] is False
    assert payload["probe_config"]["fold_tail_k_rule"] == "heldout_fold_true_rescue_count"
    assert payload["fresh_evaluation_lineage_consumed"] is False
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["promotion_claimed"] is False
    assert validate_exp279_augmentation_separability_v2_court(payload) == []
