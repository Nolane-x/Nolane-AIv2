from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_exp290_tiny_root_runner_preserves_frozen_training_and_evaluation_boundaries() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import _run_exp290_root

    result = _run_exp290_root(
        root_seed="20260913-exp290-tiny-root",
        train_replicates=2,
        eval_replicates=2,
        eval_start_replicate=10000,
    )

    assert result["training"]["rng_stream"] == "augmentation"
    assert result["training"]["replicates"] == 2
    assert result["training"]["evaluation_lineage_consumed"] is False
    assert result["training"]["evaluation_correspondence_used_for_training"] is False
    assert result["evaluation"]["rng_stream"] == "evaluation"
    assert result["evaluation"]["replicates"] == 2
    assert result["evaluation"]["pair_count"] == 16
    assert result["post_training_model_digest"] == result["post_evaluation_model_digest"]
    assert len(result["post_training_model_digest"]) == 64
    assert set(result["evaluation"]["mode_model_digests"]) == {
        "NULL_TRANSFER_CONTROL",
        "RAW_SURFACE_TRANSFER_CONTROL",
        "LEARNED_STRUCTURAL_TRANSFER",
        "ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND",
    }
    assert len(set(result["evaluation"]["mode_model_digests"].values())) == 1
    assert result["evaluation"]["raw_surface_transfer_hit_count"] == 0
    assert result["evaluation"]["surface_namespaces_disjoint"] is True
    assert result["evaluation"]["learned_oracle_correspondence_delivered"] is False
    assert result["evaluation"]["learned_evaluator_validity_truth_delivered"] is False
    assert result["evaluation"]["learned_target_clause_truth_delivered"] is False
    assert result["target_local_clause_learning_enabled"] is False
    assert result["scientific_evidence_eligible"] is False
    assert result["confirmatory_data_consumed"] is False
    assert result["challenge_materialized"] is False
    assert result["promotion_claimed"] is False


def test_exp290_tiny_root_metrics_recompute_decision_and_have_matched_cost_identity() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import (
        _run_exp290_root,
        classify_exp290_root,
    )

    result = _run_exp290_root(
        root_seed="20260913-exp290-tiny-root-metrics",
        train_replicates=2,
        eval_replicates=2,
        eval_start_replicate=10000,
    )
    metrics = result["root_metrics"]
    assert result["decision"] == classify_exp290_root(metrics)
    assert metrics["null_cost"] > 0.0
    assert metrics["raw_cost"] > 0.0
    assert metrics["learned_cost"] > 0.0
    assert metrics["oracle_cost"] > 0.0
    assert metrics["oracle_headroom"] == pytest.approx(
        metrics["null_cost"] - metrics["oracle_cost"]
    )
    assert metrics["learned_headroom"] == pytest.approx(
        metrics["null_cost"] - metrics["learned_cost"]
    )
    assert metrics["raw_surface_transfer_hit_count"] == 0
    assert 0.0 <= metrics["learned_source_to_target_correspondence_accuracy"] <= 1.0
    assert 0.0 <= metrics["learned_exact_transferred_clause_recovery_rate"] <= 1.0
