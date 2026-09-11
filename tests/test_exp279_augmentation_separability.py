from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_augmentation_separability import (
    PROBE_KINDS,
    _assign_probe_fold,
    _average_precision,
    _economic_break_even_probability,
    _roc_auc,
    _top_k_precision,
    run_exp279_augmentation_separability_court,
    validate_exp279_augmentation_separability_court,
)


def test_probe_fold_assignment_keeps_all_three_strata_inside_each_fold_block() -> None:
    assert [_assign_probe_fold(replicate, folds=3) for replicate in range(9)] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
    ]


def test_rank_metrics_reward_perfect_rescue_ordering() -> None:
    labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float64)

    assert _roc_auc(scores, labels) == pytest.approx(1.0, abs=1e-12)
    assert _average_precision(scores, labels) == pytest.approx(1.0, abs=1e-12)
    assert _top_k_precision(scores, labels, k=2) == pytest.approx(1.0, abs=1e-12)


def test_rank_metrics_return_none_when_class_support_is_missing() -> None:
    labels = torch.zeros(5, dtype=torch.long)
    scores = torch.arange(5, dtype=torch.float64)

    assert _roc_auc(scores, labels) is None
    assert _average_precision(scores, labels) is None
    assert _top_k_precision(scores, labels, k=0) is None


def test_economic_break_even_probability_is_derived_only_from_sealed_flop_ledger() -> None:
    stop_flops = 110_046.0
    branch_flops = 224_766.0
    incremental_ratio = (branch_flops - stop_flops) / branch_flops
    expected = incremental_ratio / (1.0 + incremental_ratio)

    value = _economic_break_even_probability(
        stop_accounted_flops_per_episode=stop_flops,
        branch_accounted_flops_per_episode=branch_flops,
    )

    assert value == pytest.approx(expected, abs=1e-15)
    assert 0.0 < value < 0.5


def test_probe_family_is_fixed_before_any_probe_data_are_seen() -> None:
    assert PROBE_KINDS == (
        "LINEAR_MEAN",
        "MLP_MEAN",
        "DEEPSETS",
    )


def test_tiny_court_never_uses_evaluation_stream_and_emits_fail_closed_receipts() -> None:
    payload = run_exp279_augmentation_separability_court(
        root_seed="exp279-augmentation-separability-test",
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
        probe_steps=2,
        probe_lr=1e-2,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    assert payload["schema"] == "NLM-EXP-279-AUGMENTATION-SEPARABILITY-COURT-V1"
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["scientific_evidence_eligible"] is False
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["promotion_claimed"] is False
    assert payload["data_boundary"]["training_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["probe_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["evaluation_rng_stream_used"] is False
    assert payload["data_boundary"]["evaluation_targets_used"] is False
    assert payload["data_boundary"]["probe_root_seed"] != payload["root_seed"]
    assert payload["probe_config"]["folds"] == 3
    assert payload["probe_config"]["k_rule"] == "pooled_true_rescue_count"
    assert payload["probe_config"]["probe_kinds"] == list(PROBE_KINDS)
    assert set(payload["probes"]) == set(PROBE_KINDS)
    assert payload["court_classification"] in {
        "INSUFFICIENT_RESCUE_SUPPORT",
        "LINEAR_TAIL_SEPARABLE",
        "RICH_HEAD_TAIL_SEPARABLE",
        "REPRESENTATION_NOT_ECONOMICALLY_SEPARABLE",
    }
    assert validate_exp279_augmentation_separability_court(payload) == []


def test_survival_rule_is_preregistered_as_economic_tail_not_threshold_tuning() -> None:
    payload = run_exp279_augmentation_separability_court(
        root_seed="exp279-augmentation-separability-rule-test",
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

    rule = payload["survival_rule"]
    assert rule["minimum_pooled_roc_auc"] == pytest.approx(0.75)
    assert rule["tail_precision_threshold_source"] == "sealed_compute_ledger_break_even"
    assert rule["tail_k_rule"] == "pooled_true_rescue_count"
    assert rule["requires_all_folds_positive_and_negative_support"] is True
    assert rule["decision_threshold_tuned"] is False
    assert math.isfinite(float(rule["economic_break_even_probability"]))
