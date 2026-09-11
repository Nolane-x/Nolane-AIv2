from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_paired_runner import (
    _build_seeded_triplet,
    _hybrid_stop_decision_logits,
    run_exp279_paired_development,
)
from nolane_ai.experiments.exp279_routing_marginal import (
    _forced_stop_output,
    _routing_outcome_counts,
    run_exp279_routing_marginal_development,
)


def _logits(predictions: list[list[int]]) -> torch.Tensor:
    rows: list[list[list[float]]] = []
    for episode in predictions:
        rows.append(
            [[8.0, -8.0] if prediction == 0 else [-8.0, 8.0] for prediction in episode]
        )
    return torch.tensor(rows, dtype=torch.float32)


def test_forced_stop_output_matches_canonical_stop_logits_and_never_routes() -> None:
    _, _, hybrid, _ = _build_seeded_triplet(
        root_seed="exp279-shadow-stop-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.0,
    )
    surface = torch.randn(3, 4, 8)
    variables = torch.randn(3, 5, 8)
    incidence = torch.ones(3, 3, 5)

    shadow = _forced_stop_output(
        hybrid,
        surface_events=surface,
        variable_states=variables,
        incidence=incidence,
    )
    canonical_logits = _hybrid_stop_decision_logits(
        hybrid,
        surface_events=surface,
        variable_states=variables,
        incidence=incidence,
    )

    assert torch.allclose(shadow.decision_logits, canonical_logits, atol=0.0, rtol=0.0)
    assert shadow.branch_route_mask.tolist() == [False, False, False]
    assert shadow.representation_semantics == "hybrid_forced_stop_shadow_same_weights_no_branch_execution"


def test_routing_outcome_counts_close_rescue_harm_and_unchanged_cases() -> None:
    targets = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ],
        dtype=torch.long,
    )
    stop = _logits(
        [
            [1, 0],  # routed stop fail -> actual rescue
            [0, 1],  # routed stop success -> actual harm
            [1, 1],  # routed success -> success
            [0, 1],  # routed fail -> fail
            [0, 0],  # unrouted exact success
        ]
    )
    actual = _logits(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 0],
        ]
    )
    route_mask = torch.tensor([True, True, True, True, False])

    counts = _routing_outcome_counts(
        stop_logits=stop,
        actual_logits=actual,
        targets=targets,
        route_mask=route_mask,
    )

    assert counts == {
        "episodes": 5,
        "routed_episodes": 4,
        "unrouted_episodes": 1,
        "routed_rescues": 1,
        "routed_harms": 1,
        "routed_success_unchanged": 1,
        "routed_failure_unchanged": 1,
        "net_routed_exact_delta": 0,
    }


def test_shadow_runner_replays_canonical_hybrid_training_exactly() -> None:
    kwargs = dict(
        root_seed="exp279-shadow-replay-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        train_replicates=3,
        eval_replicates=3,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    canonical = run_exp279_paired_development(**kwargs)
    shadow = run_exp279_routing_marginal_development(**kwargs)

    assert shadow["schema"] == "NLM-EXP-279-ROUTING-MARGINAL-DEV-V1"
    assert shadow["evidence_level"] == "EV-E2"
    assert shadow["decision"] == "UNVERIFIED"
    assert shadow["scientific_evidence_eligible"] is False
    assert shadow["final_hybrid_digest"] == canonical["final_state"]["hybrid_digest"]
    assert shadow["training_batch_digests"] == canonical["training"]["paired_batch_digests"]
    assert shadow["challenge_materialized"] is False
    assert shadow["confirmatory_data_consumed"] is False
    assert shadow["promotion_claimed"] is False


def test_shadow_runner_classifies_threshold_one_as_routing_inactive() -> None:
    receipt = run_exp279_routing_marginal_development(
        root_seed="exp279-shadow-inactive-test",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=1.0,
        train_replicates=3,
        eval_replicates=3,
        eval_start_replicate=100,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    aggregate = receipt["aggregate"]
    assert aggregate["total_routed_episodes"] == 0
    assert aggregate["routing_classification"] == "ROUTING_INACTIVE"
    assert aggregate["mean_actual_solution_rate"] == pytest.approx(
        aggregate["mean_forced_stop_solution_rate"]
    )
    assert aggregate["mean_actual_utility"] == pytest.approx(
        aggregate["mean_forced_stop_utility"]
    )
    assert aggregate["routing_marginal_utility_gain"] == pytest.approx(0.0)
