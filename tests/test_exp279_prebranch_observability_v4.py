from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import _build_seeded_triplet
from nolane_ai.experiments.matched_routing_arms import HybridRoutingArm
from nolane_ai.experiments.exp279_prebranch_observability_v4 import (
    PROBE_KINDS,
    PROBE_ROOT_SUFFIX,
    SCHEMA,
    _aggregate_fold_direct_utility_metrics,
    _build_probe,
    _fit_probe_fold,
    _fold_direct_utility_metrics,
    _pairwise_ranking_loss,
    _prebranch_features,
    _state_event_residual_vector,
    run_exp279_prebranch_observability_v4_court,
)


def test_pairwise_probe_loss_remains_shift_invariant() -> None:
    scores = torch.tensor([2.0, -1.0, 0.5, 1.5])
    labels = torch.tensor([1, 0, 0, 1])
    first = _pairwise_ranking_loss(scores, labels)
    second = _pairwise_ranking_loss(scores + 101.25, labels)
    assert first.item() == pytest.approx(second.item(), abs=1e-6)


def test_prebranch_features_use_exact_cheap_observables_without_branch_gru(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-v4-feature-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
    )

    def forbidden_branch_context(self: HybridRoutingArm, events: torch.Tensor) -> torch.Tensor:
        raise AssertionError("branch GRU context must not be used for V4 probe features")

    monkeypatch.setattr(HybridRoutingArm, "_branch_context", forbidden_branch_context)
    surface_events = torch.randn(3, 4, 8)
    variable_states = torch.randn(3, 5, 8)
    incidence = torch.ones(3, 2, 5)
    features = _prebranch_features(
        hybrid,
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
    )

    assert set(features) == {"propagation_state", "p_mean", "e_mean", "e_delta"}
    assert features["propagation_state"].shape == (3, 5, 6)
    assert features["p_mean"].shape == (3, 6)
    assert features["e_mean"].shape == (3, 6)
    assert features["e_delta"].shape == (3, 6)


def test_probe_family_is_frozen_and_residual_vector_has_exact_semantics() -> None:
    assert PROBE_KINDS == ("STATE_DEEPSETS", "STATE_EVENT_MEAN", "STATE_EVENT_RESIDUAL")
    batch, variables, hidden = 4, 3, 5
    propagation_state = torch.randn(batch, variables, hidden)
    p_mean = propagation_state.mean(dim=1)
    e_mean = torch.randn(batch, hidden)
    e_delta = torch.randn(batch, hidden)
    features = {
        "propagation_state": propagation_state,
        "p_mean": p_mean,
        "e_mean": e_mean,
        "e_delta": e_delta,
    }
    residual = _state_event_residual_vector(features)
    expected = torch.cat((p_mean, e_mean, e_delta, p_mean - e_mean, p_mean * e_mean), dim=-1)
    assert residual.shape == (batch, 5 * hidden)
    assert torch.equal(residual, expected)

    for offset, kind in enumerate(PROBE_KINDS):
        probe = _build_probe(kind, hidden_size=hidden, seed=123 + offset)
        if kind == "STATE_DEEPSETS":
            scores = probe(features["propagation_state"])
        elif kind == "STATE_EVENT_MEAN":
            scores = probe(torch.cat((p_mean, e_mean), dim=-1))
        else:
            scores = probe(residual)
        assert scores.shape == (batch,)


def test_fold_direct_utility_routes_top_k_and_charges_exact_paths() -> None:
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


def test_cross_fold_aggregate_uses_total_counts_and_flops() -> None:
    base = {
        "support_closed": True,
        "roc_auc": 0.5,
        "average_precision": 0.5,
        "concordance_pair_count": 3,
    }
    rows = [
        base
        | {
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
        base
        | {
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


def test_fit_side_support_is_required_even_when_heldout_support_exists() -> None:
    hidden = 4
    features = {
        "propagation_state": torch.randn(6, 2, hidden),
        "p_mean": torch.randn(6, hidden),
        "e_mean": torch.randn(6, hidden),
        "e_delta": torch.randn(6, hidden),
    }
    fold_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    # Held-out fold 0 has one rescue and one non-rescue. Fit folds have no rescue.
    stop_exact = torch.tensor([False, True, True, False, True, False])
    branch_exact = torch.tensor([True, True, True, False, True, False])
    row = _fit_probe_fold(
        kind="STATE_EVENT_MEAN",
        features=features,
        stop_exact=stop_exact,
        branch_exact=branch_exact,
        fold_ids=fold_ids,
        fold=0,
        folds=3,
        hidden_size=hidden,
        probe_steps=1,
        probe_lr=1e-2,
        probe_root_seed="v4-support-test",
        stop_cost=10,
        branch_cost=20,
    )
    assert row["fit_support_closed"] is False
    assert row["heldout_support_closed"] is True
    assert row["support_closed"] is False
    assert row["direct_utility_improved"] is False


def test_tiny_v4_court_preserves_development_and_probe_economics_boundary() -> None:
    payload = run_exp279_prebranch_observability_v4_court(
        root_seed="exp279-prebranch-v4-tiny-test",
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
    assert payload["feature_boundary"]["branch_gru_used_for_probe_features"] is False
    assert payload["feature_boundary"]["allowed_temporal_summary"] == "last_projected_event_minus_first_projected_event"
    assert payload["probe_config"]["probe_kinds"] == list(PROBE_KINDS)
    assert payload["probe_config"]["raw_scores_compared_across_folds"] is False
    assert payload["probe_economics"]["probe_is_deployable"] is False
    assert payload["probe_economics"]["probe_inference_flops_charged_to_primary_utility"] is False
    assert payload["probe_economics"]["successor_must_account_selector_flops"] is True
    assert payload["economic_rule"]["metric"] == "direct_counterfactual_verified_solution_per_total_accounted_flop"
    assert payload["economic_rule"]["posterior_break_even_threshold_used"] is False
    assert payload["fresh_evaluation_lineage_may_be_reserved"] is False
    assert payload["fresh_evaluation_lineage_consumed"] is False
    assert payload["confirmatory_data_consumed"] is False
    assert payload["challenge_materialized"] is False
    assert payload["promotion_claimed"] is False
    assert isinstance(payload["artifact_digest"], str) and len(payload["artifact_digest"]) == 64
    assert math.isfinite(float(payload["compute_economics"]["stop_accounted_flops_per_episode"]))
