from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_branch_rescue_oracle import (
    _branch_outcome_counts,
    _rescue_oracle_batch_metrics,
    run_exp279_branch_rescue_oracle_development,
)
from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development


def _logits(predictions: list[list[int]]) -> torch.Tensor:
    return torch.tensor(
        [
            [[8.0, -8.0] if value == 0 else [-8.0, 8.0] for value in episode]
            for episode in predictions
        ],
        dtype=torch.float32,
    )


def test_branch_outcome_partition_closes_rescues_harms_successes_and_failures() -> None:
    targets = torch.tensor(
        [[0, 0], [0, 1], [1, 1], [1, 0]],
        dtype=torch.long,
    )
    stop = _logits(
        [
            [1, 0],  # stop fails, branch rescues
            [0, 1],  # stop succeeds, branch harms
            [1, 1],  # both succeed
            [0, 1],  # both fail
        ]
    )
    branch = _logits(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
        ]
    )

    counts = _branch_outcome_counts(stop_logits=stop, branch_logits=branch, targets=targets)

    assert counts == {
        "episodes": 4,
        "stop_exact_successes": 2,
        "forced_branch_exact_successes": 2,
        "branch_rescues": 1,
        "branch_harms": 1,
        "both_success": 1,
        "both_failure": 1,
    }


def test_rescue_oracle_can_be_cost_effective_when_rescues_are_cheap_enough() -> None:
    targets = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)
    stop = _logits([[0], [1], [1], [1]])
    branch = _logits([[1], [0], [1], [1]])

    metrics = _rescue_oracle_batch_metrics(
        stop_logits=stop,
        branch_logits=branch,
        targets=targets,
        stop_accounted_flops_per_episode=100.0,
        branch_accounted_flops_per_episode=200.0,
    )

    assert metrics["branch_rescues"] == 1
    assert metrics["branch_harms"] == 1
    assert metrics["rescue_oracle_route_fraction"] == pytest.approx(0.25)
    assert metrics["stop_solution_rate"] == pytest.approx(0.25)
    assert metrics["rescue_oracle_solution_rate"] == pytest.approx(0.50)
    assert metrics["rescue_oracle_accounted_flops_per_episode"] == pytest.approx(125.0)
    assert metrics["stop_utility"] == pytest.approx(0.0025)
    assert metrics["rescue_oracle_utility"] == pytest.approx(0.004)
    assert metrics["rescue_oracle_relative_utility_gain"] == pytest.approx(0.6)


def test_rescue_oracle_can_expose_rescues_that_are_not_cost_effective() -> None:
    targets = torch.tensor([[0], [0], [0], [0]], dtype=torch.long)
    stop = _logits([[0], [1], [1], [1]])
    branch = _logits([[1], [0], [1], [1]])

    metrics = _rescue_oracle_batch_metrics(
        stop_logits=stop,
        branch_logits=branch,
        targets=targets,
        stop_accounted_flops_per_episode=100.0,
        branch_accounted_flops_per_episode=1_000.0,
    )

    assert metrics["branch_rescues"] == 1
    assert metrics["rescue_oracle_accounted_flops_per_episode"] == pytest.approx(325.0)
    assert metrics["rescue_oracle_utility"] < metrics["stop_utility"]
    assert metrics["rescue_oracle_relative_utility_gain"] < 0.0


def test_branch_rescue_runner_replays_canonical_hybrid_training_exactly() -> None:
    kwargs = dict(
        root_seed="exp279-branch-rescue-replay",
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
    diagnostic = run_exp279_branch_rescue_oracle_development(**kwargs)

    assert diagnostic["schema"] == "NLM-EXP-279-BRANCH-RESCUE-ORACLE-DEV-V1"
    assert diagnostic["evidence_level"] == "EV-E2"
    assert diagnostic["decision"] == "UNVERIFIED"
    assert diagnostic["scientific_evidence_eligible"] is False
    assert diagnostic["final_hybrid_digest"] == canonical["final_state"]["hybrid_digest"]
    assert diagnostic["training_batch_digests"] == canonical["training"]["paired_batch_digests"]
    assert diagnostic["challenge_materialized"] is False
    assert diagnostic["confirmatory_data_consumed"] is False
    assert diagnostic["promotion_claimed"] is False


def test_branch_rescue_classification_has_exact_three_way_contract() -> None:
    receipt = run_exp279_branch_rescue_oracle_development(
        root_seed="exp279-branch-rescue-classification",
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
    assert receipt["aggregate"]["branch_rescue_classification"] in {
        "NO_BRANCH_RESCUE_CAPACITY",
        "RESCUES_EXIST_BUT_ORACLE_NOT_COST_EFFECTIVE",
        "BRANCH_RESCUE_CAPACITY_COST_EFFECTIVE",
    }
