from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _build_seeded_triplet
from nolane_ai.experiments.exp279_cost_accounted_branch_preview_v5 import (
    PREVIEW_FAMILIES,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    _aggregate_fold_preview_direct_utility_metrics,
    _fold_preview_direct_utility_metrics,
    _pairwise_ranking_loss,
    _preview_hidden,
    _resume_branch_hidden,
    _selector_costs,
    run_exp279_cost_accounted_branch_preview_v5_court,
)


def test_preview_family_is_frozen() -> None:
    assert PREVIEW_FAMILIES == (
        "PREFIX1_STATE_LINEAR",
        "PREFIX2_STATE_LINEAR",
        "PREFIX3_STATE_LINEAR",
    )
    assert PROBE_ROOT_SUFFIX == "::independent-augmentation-preview-v5"


def test_pairwise_loss_is_shift_invariant() -> None:
    scores = torch.tensor([2.0, -1.0, 0.5, 1.5])
    labels = torch.tensor([1, 0, 0, 1])
    a = _pairwise_ranking_loss(scores, labels)
    b = _pairwise_ranking_loss(scores + 99.0, labels)
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_prefix_plus_suffix_resume_matches_full_branch_hidden() -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-v5-resume-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
    )
    surface_events = torch.randn(4, 4, 8)
    variable_states = torch.randn(4, 5, 8)
    events, _ = hybrid._validate_common(surface_events, variable_states)

    with torch.no_grad():
        _, full_hidden = hybrid.branch_gru(events)
        for depth in (1, 2, 3):
            prefix_hidden = _preview_hidden(hybrid, events, depth=depth)
            resumed = _resume_branch_hidden(
                hybrid,
                events,
                depth=depth,
                prefix_hidden=prefix_hidden,
            )
            assert torch.allclose(full_hidden, resumed, atol=1e-6, rtol=1e-6)


def test_selector_costs_charge_pool_and_linear_head() -> None:
    costs = _selector_costs(hidden_size=48, variables=6, preview_depth=2)
    assert costs["preview_gru_flops"] == 2 * (12 * 48 * 48 + 20 * 48)
    assert costs["p_mean_flops"] == 6 * 48
    assert costs["selector_head_flops"] == 4 * 48 + 1
    assert costs["selector_total_flops"] == 6 * 48 + 4 * 48 + 1


def test_fold_preview_utility_charges_unselected_preview_but_not_selected_twice() -> None:
    scores = torch.tensor([0.1, 3.0, -2.0, 0.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_preview_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=30,
        preview_gru_flops_per_unselected_episode=5,
        selector_flops_per_episode=2,
    )
    assert row["support_closed"] is True
    assert row["rescue_count"] == 1
    assert row["route_k"] == 1
    assert row["selected_rescues"] == 1
    assert row["selected_harms"] == 0
    assert row["stop_solution_count"] == 2
    assert row["routed_solution_count"] == 3
    assert row["stop_total_accounted_flops"] == 40
    assert row["preview_routed_total_accounted_flops"] == 3 * (10 + 5 + 2) + (30 + 2)
    assert row["stop_verified_utility"] == pytest.approx(2 / 40)
    assert row["preview_routed_verified_utility"] == pytest.approx(3 / 83)
    assert row["direct_utility_improved"] is False
    assert row["raw_scores_exported"] is False


def test_fold_preview_utility_counts_harms() -> None:
    scores = torch.tensor([0.0, 0.1, -1.0, 5.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_preview_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=30,
        preview_gru_flops_per_unselected_episode=1,
        selector_flops_per_episode=1,
    )
    assert row["route_k"] == 1
    assert row["selected_rescues"] == 0
    assert row["selected_harms"] == 1
    assert row["routed_solution_count"] == 1
    assert row["direct_utility_improved"] is False


def test_aggregate_uses_total_counts_and_preview_flops() -> None:
    rows = [
        {
            "support_closed": True,
            "stop_solution_count": 2,
            "routed_solution_count": 3,
            "stop_total_accounted_flops": 40.0,
            "preview_routed_total_accounted_flops": 83.0,
            "route_k": 1,
            "selected_rescues": 1,
            "selected_harms": 0,
            "selected_neutral": 0,
            "rescue_count": 1,
            "episode_count": 4,
            "roc_auc": 0.5,
            "average_precision": 0.5,
            "concordance_pair_count": 3,
        },
        {
            "support_closed": True,
            "stop_solution_count": 8,
            "routed_solution_count": 8,
            "stop_total_accounted_flops": 100.0,
            "preview_routed_total_accounted_flops": 120.0,
            "route_k": 1,
            "selected_rescues": 0,
            "selected_harms": 0,
            "selected_neutral": 1,
            "rescue_count": 1,
            "episode_count": 10,
            "roc_auc": 0.5,
            "average_precision": 0.5,
            "concordance_pair_count": 9,
        },
    ]
    aggregate = _aggregate_fold_preview_direct_utility_metrics(rows)
    assert aggregate["stop_solution_count"] == 10
    assert aggregate["routed_solution_count"] == 11
    assert aggregate["stop_total_accounted_flops"] == 140.0
    assert aggregate["preview_routed_total_accounted_flops"] == 203.0
    assert aggregate["stop_verified_utility"] == pytest.approx(10 / 140)
    assert aggregate["preview_routed_verified_utility"] == pytest.approx(11 / 203)
    assert aggregate["direct_utility_improved"] is False
    assert aggregate["raw_scores_compared_across_folds"] is False


def test_tiny_court_preserves_boundary_and_charges_preview_costs() -> None:
    artifact = run_exp279_cost_accounted_branch_preview_v5_court(
        root_seed="20260911-exp279-v5-tiny-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        probe_replicates=9,
        probe_folds=3,
        batch_size=2,
        timesteps=4,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=2e-3,
        weight_decay=0.0,
        probe_steps=1,
        probe_lr=1e-2,
        protocol_digest="c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440",
        code_digest="test-code-digest",
    )
    assert artifact["schema"] == SCHEMA
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["data_boundary"]["training_rng_stream"] == "augmentation"
    assert artifact["data_boundary"]["probe_rng_stream"] == "augmentation"
    assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
    assert artifact["data_boundary"]["evaluation_targets_used"] is False
    assert artifact["preview_cost_rule"]["preview_flops_charged_to_primary_utility"] is True
    assert artifact["preview_cost_rule"]["selector_flops_charged_to_primary_utility"] is True
    assert artifact["preview_semantics"]["preview_hidden_reused_for_selected_branch"] is True
    assert artifact["probe_config"]["raw_scores_compared_across_folds"] is False
    assert artifact["probe_config"]["raw_scores_exported"] is False
    assert artifact["fresh_evaluation_lineage_may_be_reserved"] is False
    assert artifact["fresh_evaluation_lineage_consumed"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["promotion_claimed"] is False
    assert artifact["scientific_evidence_eligible"] is False
