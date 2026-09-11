from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_augmentation_direct_utility_v3 import (
    PROBE_KINDS,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    _aggregate_fold_direct_utility_metrics,
    _fold_direct_utility_metrics,
    _pairwise_ranking_loss,
    run_exp279_augmentation_direct_utility_v3_court,
)


def test_pairwise_probe_loss_remains_shift_invariant() -> None:
    scores = torch.tensor([2.0, -1.0, 0.5, 1.5])
    labels = torch.tensor([1, 0, 0, 1])
    first = _pairwise_ranking_loss(scores, labels)
    second = _pairwise_ranking_loss(scores + 101.25, labels)
    assert first.item() == pytest.approx(second.item(), abs=1e-6)


def test_fold_direct_utility_routes_top_k_and_charges_exact_paths() -> None:
    # Only episode 1 is an exact rescue, so oracle diagnostic cardinality k=1.
    scores = torch.tensor([0.1, 3.0, -2.0, 0.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=20,
    )

    assert row["support_closed"] is True
    assert row["rescue_count"] == 1
    assert row["route_k"] == 1
    assert row["selected_rescues"] == 1
    assert row["selected_harms"] == 0
    assert row["stop_solution_count"] == 2
    assert row["routed_solution_count"] == 3
    assert row["stop_total_accounted_flops"] == 40
    assert row["routed_total_accounted_flops"] == 50
    assert row["stop_verified_utility"] == pytest.approx(2 / 40)
    assert row["routed_verified_utility"] == pytest.approx(3 / 50)
    assert row["direct_utility_improved"] is True
    assert row["raw_scores_exported"] is False


def test_fold_direct_utility_counts_harm_and_can_reject_bad_ranking() -> None:
    scores = torch.tensor([0.0, 0.1, -1.0, 5.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=20,
    )

    assert row["route_k"] == 1
    assert row["selected_rescues"] == 0
    assert row["selected_harms"] == 1
    assert row["routed_solution_count"] == 1
    assert row["direct_utility_improved"] is False


def test_fold_selection_is_invariant_to_score_offset() -> None:
    scores = torch.tensor([0.2, 3.5, -0.4, 0.1])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    kwargs = dict(
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=20,
    )
    first = _fold_direct_utility_metrics(scores, stop_exact, branch_exact, **kwargs)
    second = _fold_direct_utility_metrics(scores + 1000.0, stop_exact, branch_exact, **kwargs)
    comparable = (
        "route_k",
        "selected_rescues",
        "selected_harms",
        "routed_solution_count",
        "routed_total_accounted_flops",
        "direct_utility_improved",
    )
    assert {key: first[key] for key in comparable} == {key: second[key] for key in comparable}


def test_cross_fold_aggregate_sums_solution_and_flop_sufficient_statistics() -> None:
    rows = [
        {
            "support_closed": True,
            "stop_solution_count": 2,
            "routed_solution_count": 3,
            "stop_total_accounted_flops": 40,
            "routed_total_accounted_flops": 50,
            "route_k": 1,
            "selected_rescues": 1,
            "selected_harms": 0,
            "selected_neutral": 0,
            "rescue_count": 1,
            "episode_count": 4,
        },
        {
            "support_closed": True,
            "stop_solution_count": 8,
            "routed_solution_count": 8,
            "stop_total_accounted_flops": 100,
            "routed_total_accounted_flops": 110,
            "route_k": 1,
            "selected_rescues": 0,
            "selected_harms": 0,
            "selected_neutral": 1,
            "rescue_count": 1,
            "episode_count": 10,
        },
    ]
    aggregate = _aggregate_fold_direct_utility_metrics(rows)
    assert aggregate["stop_solution_count"] == 10
    assert aggregate["routed_solution_count"] == 11
    assert aggregate["stop_total_accounted_flops"] == 140
    assert aggregate["routed_total_accounted_flops"] == 160
    assert aggregate["stop_verified_utility"] == pytest.approx(10 / 140)
    assert aggregate["routed_verified_utility"] == pytest.approx(11 / 160)
    assert aggregate["direct_utility_improved"] is False
    assert aggregate["raw_scores_compared_across_folds"] is False


def test_missing_rescue_support_fails_closed() -> None:
    row = _fold_direct_utility_metrics(
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
        torch.tensor([True, False]),
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=20,
    )
    assert row["rescue_count"] == 0
    assert row["support_closed"] is False
    assert row["route_k"] == 0
    assert row["direct_utility_improved"] is False


def test_tiny_v3_court_preserves_development_boundary() -> None:
    payload = run_exp279_augmentation_direct_utility_v3_court(
        root_seed="exp279-direct-utility-v3-tiny-test",
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

    assert payload["schema"] == SCHEMA
    assert payload["evidence_level"] == "EV-E2"
    assert payload["decision"] == "UNVERIFIED"
    assert payload["scientific_evidence_eligible"] is False
    assert payload["data_boundary"]["training_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["probe_rng_stream"] == "augmentation"
    assert payload["data_boundary"]["probe_root_seed"].endswith(PROBE_ROOT_SUFFIX)
    assert payload["data_boundary"]["evaluation_rng_stream_used"] is False
    assert payload["data_boundary"]["evaluation_targets_used"] is False
    assert payload["probe_config"]["probe_kinds"] == list(PROBE_KINDS)
    assert payload["probe_config"]["raw_scores_compared_across_folds"] is False
    assert payload["probe_config"]["oracle_route_cardinality_is_deployable"] is False
    assert payload["economic_rule"]["metric"] == "direct_counterfactual_verified_solution_per_total_accounted_flop"
    assert payload["economic_rule"]["posterior_break_even_threshold_used"] is False
    assert payload["fresh_evaluation_lineage_consumed"] is False
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["promotion_claimed"] is False
    assert isinstance(payload["artifact_digest"], str) and len(payload["artifact_digest"]) == 64
    assert math.isfinite(float(payload["compute_economics"]["stop_accounted_flops_per_episode"]))
