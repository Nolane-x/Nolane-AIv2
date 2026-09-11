from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _build_seeded_triplet
from nolane_ai.experiments.exp279_counterfactual_delta_distill_v6 import (
    CONTROL_FAMILY,
    PRIMARY_FAMILY,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    _aggregate_fold_direct_utility_metrics,
    _cdd_costs,
    _cheap_features,
    _fold_direct_utility_metrics,
    _pairwise_ranking_loss,
    _stop_and_branch_states,
    _teacher_delta,
    run_exp279_counterfactual_delta_distill_v6_court,
)


def _tiny_hybrid():
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-v6-core-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
    )
    return hybrid


def test_v6_primary_and_control_families_are_frozen() -> None:
    assert PRIMARY_FAMILY == "CDD_DELTA_LINEAR"
    assert CONTROL_FAMILY == "RAW_CHEAP_LINEAR"
    assert PROBE_ROOT_SUFFIX == "::independent-augmentation-cdd-v6"


def test_pairwise_ranking_loss_is_shift_invariant() -> None:
    scores = torch.tensor([2.0, -1.0, 0.5, 1.5])
    labels = torch.tensor([1, 0, 0, 1])
    a = _pairwise_ranking_loss(scores, labels)
    b = _pairwise_ranking_loss(scores + 137.0, labels)
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_cheap_features_are_prebranch_and_have_frozen_shape() -> None:
    hybrid = _tiny_hybrid()
    surface_events = torch.randn(4, 4, 8)
    variable_states = torch.randn(4, 5, 8)
    incidence = torch.ones(4, 2, 5)

    cheap, p_mean = _cheap_features(
        hybrid,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    assert cheap.shape == (4, 3 * hybrid.hidden_size)
    assert p_mean.shape == (4, hybrid.hidden_size)
    assert torch.isfinite(cheap).all()
    assert torch.isfinite(p_mean).all()


def test_teacher_delta_is_detached_full_branch_minus_stop_mean() -> None:
    hybrid = _tiny_hybrid()
    surface_events = torch.randn(3, 4, 8)
    variable_states = torch.randn(3, 5, 8)
    incidence = torch.ones(3, 2, 5)

    stop_state, branch_state = _stop_and_branch_states(
        hybrid,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )
    teacher = _teacher_delta(stop_state, branch_state)
    expected = (branch_state - stop_state).mean(dim=1).detach()
    assert teacher.shape == (3, hybrid.hidden_size)
    assert torch.allclose(teacher, expected, atol=1e-7, rtol=1e-7)
    assert teacher.requires_grad is False


def test_cdd_costs_charge_summaries_distiller_and_selector() -> None:
    hidden = 4
    variables = 3
    timesteps = 5
    costs = _cdd_costs(hidden_size=hidden, variables=variables, timesteps=timesteps)
    expected_summary = variables * hidden + timesteps * hidden + hidden
    expected_distiller = (
        (2 * 3 * hidden * hidden + hidden)
        + 4 * hidden
        + (2 * hidden * hidden + hidden)
    )
    expected_selector = 4 * hidden + 1
    assert costs["cheap_summary_flops"] == expected_summary
    assert costs["distiller_flops"] == expected_distiller
    assert costs["selector_flops"] == expected_selector
    assert costs["total_cdd_inference_flops"] == expected_summary + expected_distiller + expected_selector


def test_fold_direct_utility_charges_cdd_on_stop_and_branch_paths() -> None:
    scores = torch.tensor([0.1, 3.0, -2.0, 0.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=30,
        cdd_inference_flops_per_episode=3,
    )
    assert row["support_closed"] is True
    assert row["rescue_count"] == 1
    assert row["route_k"] == 1
    assert row["selected_rescues"] == 1
    assert row["selected_harms"] == 0
    assert row["stop_solution_count"] == 2
    assert row["routed_solution_count"] == 3
    assert row["stop_total_accounted_flops"] == 40
    assert row["cdd_routed_total_accounted_flops"] == 3 * (10 + 3) + (30 + 3)
    assert row["stop_verified_utility"] == pytest.approx(2 / 40)
    assert row["cdd_routed_verified_utility"] == pytest.approx(3 / 72)
    assert row["direct_utility_improved"] is False
    assert row["raw_scores_exported"] is False


def test_fold_direct_utility_counts_selected_harm() -> None:
    scores = torch.tensor([0.0, 0.1, -1.0, 5.0])
    stop_exact = torch.tensor([True, False, False, True])
    branch_exact = torch.tensor([True, True, False, False])
    row = _fold_direct_utility_metrics(
        scores,
        stop_exact,
        branch_exact,
        stop_accounted_flops_per_episode=10,
        branch_accounted_flops_per_episode=30,
        cdd_inference_flops_per_episode=1,
    )
    assert row["route_k"] == 1
    assert row["selected_rescues"] == 0
    assert row["selected_harms"] == 1
    assert row["routed_solution_count"] == 1
    assert row["solution_count_delta"] == -1
    assert row["direct_utility_improved"] is False


def test_aggregate_uses_total_counts_and_total_flops() -> None:
    rows = [
        {
            "support_closed": True,
            "stop_solution_count": 2,
            "routed_solution_count": 3,
            "stop_total_accounted_flops": 40.0,
            "cdd_routed_total_accounted_flops": 72.0,
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
            "cdd_routed_total_accounted_flops": 120.0,
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
    aggregate = _aggregate_fold_direct_utility_metrics(rows)
    assert aggregate["stop_solution_count"] == 10
    assert aggregate["routed_solution_count"] == 11
    assert aggregate["stop_total_accounted_flops"] == 140.0
    assert aggregate["cdd_routed_total_accounted_flops"] == 192.0
    assert aggregate["stop_verified_utility"] == pytest.approx(10 / 140)
    assert aggregate["cdd_routed_verified_utility"] == pytest.approx(11 / 192)
    assert aggregate["raw_scores_compared_across_folds"] is False


def test_tiny_court_preserves_boundary_and_charges_cdd() -> None:
    artifact = run_exp279_counterfactual_delta_distill_v6_court(
        root_seed="20260912-exp279-v6-tiny-test",
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
        distill_steps=1,
        distill_lr=2e-3,
        selector_steps=1,
        selector_lr=1e-2,
        protocol_digest="c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440",
        code_digest="test-code-digest",
    )
    assert artifact["schema"] == SCHEMA
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["scientific_evidence_eligible"] is False
    assert artifact["canonical_model_frozen_for_cdd"] is True
    assert artifact["teacher_rule"]["teacher_uses_fit_partition_only"] is True
    assert artifact["data_boundary"]["training_rng_stream"] == "augmentation"
    assert artifact["data_boundary"]["probe_rng_stream"] == "augmentation"
    assert artifact["data_boundary"]["evaluation_rng_stream_used"] is False
    assert artifact["data_boundary"]["evaluation_targets_used"] is False
    assert artifact["data_boundary"]["heldout_branch_hidden_used_for_features"] is False
    assert artifact["cdd_cost_rule"]["cdd_inference_flops_charged_to_primary_utility"] is True
    assert artifact["teacher_rule"]["teacher_training_flops_excluded_from_deployment_utility"] is True
    assert artifact["probe_config"]["raw_scores_compared_across_folds"] is False
    assert artifact["probe_config"]["raw_scores_exported"] is False
    assert artifact["fresh_evaluation_lineage_may_be_reserved"] is False
    assert artifact["fresh_evaluation_lineage_consumed"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["promotion_claimed"] is False
