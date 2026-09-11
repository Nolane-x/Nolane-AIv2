from __future__ import annotations

from copy import deepcopy

import pytest

from nolane_ai.experiments.exp279_stop_path_ablation import (
    build_exp279_stop_path_ablation,
    build_exp279_stop_path_sweep,
    validate_exp279_stop_path_ablation,
)


def _payload(
    *,
    train_replicates: int = 15,
    propagation_solution_rates: tuple[float, ...] = (0.25, 0.25),
    hybrid_solution_rates: tuple[float, ...] = (0.25, 0.25),
    propagation_cost: float = 200.0,
    hybrid_cost: float = 100.0,
) -> dict[str, object]:
    if len(propagation_solution_rates) != len(hybrid_solution_rates):
        raise ValueError("test fixture solution-rate lengths must match")
    rows: list[dict[str, object]] = []
    for offset, (prop_solution, hybrid_solution) in enumerate(
        zip(propagation_solution_rates, hybrid_solution_rates)
    ):
        rows.append(
            {
                "replicate": 20_000 + offset,
                "stratum": ("PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL")[offset % 3],
                "hybrid_route_receipt": {
                    "routed_episodes": 0,
                    "route_fraction": 0.0,
                    "batch_size": 8,
                    "branch_route_mask": [False] * 8,
                },
                "propagation_only": {
                    "verified_solution_rate": prop_solution,
                    "accounted_flops_per_episode": propagation_cost,
                    "verified_utility_per_accounted_flop_on_structure_dense_stratum": prop_solution
                    / propagation_cost,
                },
                "hybrid": {
                    "verified_solution_rate": hybrid_solution,
                    "accounted_flops_per_episode": hybrid_cost,
                    "verified_utility_per_accounted_flop_on_structure_dense_stratum": hybrid_solution
                    / hybrid_cost,
                },
            }
        )

    mean_prop_solution = sum(propagation_solution_rates) / len(propagation_solution_rates)
    mean_hybrid_solution = sum(hybrid_solution_rates) / len(hybrid_solution_rates)
    mean_prop_utility = mean_prop_solution / propagation_cost
    mean_hybrid_utility = mean_hybrid_solution / hybrid_cost
    relative_gain = mean_hybrid_utility / mean_prop_utility - 1.0
    return {
        "schema": "NLM-EXP-279-PAIRED-DEV-EVAL-V1",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "challenge_materialized": False,
        "confirmatory_data_consumed": False,
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "protocol_digest": "p" * 64,
        "code_digest": "c" * 64,
        "artifact_digest": (hex(train_replicates)[2:] * 64)[:64],
        "training": {
            "replicates": train_replicates,
            "start_replicate": 0,
        },
        "evaluation": {
            "start_replicate": 20_000,
            "replicates": len(rows),
            "per_replicate": rows,
            "aggregate": {
                "mean_propagation_solution_rate": mean_prop_solution,
                "mean_hybrid_solution_rate": mean_hybrid_solution,
                "mean_propagation_utility": mean_prop_utility,
                "mean_hybrid_utility": mean_hybrid_utility,
                "hybrid_relative_utility_gain": relative_gain,
            },
        },
    }


def test_stop_path_ablation_reconstructs_utility_gain_from_accuracy_and_compute() -> None:
    receipt = build_exp279_stop_path_ablation(_payload())

    assert receipt["schema"] == "NLM-EXP-279-STOP-PATH-ABLATION-DEV-V1"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["all_hybrid_routes_zero"] is True
    assert receipt["propagation_accounted_flops_per_episode"] == 200.0
    assert receipt["hybrid_stop_accounted_flops_per_episode"] == 100.0
    assert receipt["accuracy_factor"] == pytest.approx(1.0)
    assert receipt["compute_factor"] == pytest.approx(2.0)
    assert receipt["reconstructed_utility_factor"] == pytest.approx(2.0)
    assert receipt["observed_utility_factor"] == pytest.approx(2.0)
    assert receipt["observed_relative_utility_gain"] == pytest.approx(1.0)
    assert receipt["equal_cost_counterfactual_relative_gain"] == pytest.approx(0.0)
    assert receipt["compute_only_counterfactual_relative_gain"] == pytest.approx(1.0)
    assert receipt["mechanism_classification"] == "COMPUTE_DOMINANT_STOP_PATH_ADVANTAGE"
    assert receipt["challenge_materialized"] is False
    assert receipt["confirmatory_data_consumed"] is False
    assert validate_exp279_stop_path_ablation(receipt) == []


def test_stop_path_ablation_can_show_compute_overcoming_negative_accuracy_effect() -> None:
    receipt = build_exp279_stop_path_ablation(
        _payload(
            propagation_solution_rates=(0.50, 0.50),
            hybrid_solution_rates=(0.45, 0.45),
            propagation_cost=200.0,
            hybrid_cost=100.0,
        )
    )

    assert receipt["accuracy_factor"] == pytest.approx(0.9)
    assert receipt["compute_factor"] == pytest.approx(2.0)
    assert receipt["observed_utility_factor"] == pytest.approx(1.8)
    assert receipt["observed_relative_utility_gain"] == pytest.approx(0.8)
    assert receipt["equal_cost_counterfactual_relative_gain"] == pytest.approx(-0.1)
    assert receipt["mechanism_classification"] == "COMPUTE_DOMINANT_STOP_PATH_ADVANTAGE"


def test_stop_path_ablation_rejects_any_dynamic_routing() -> None:
    payload = _payload()
    row = payload["evaluation"]["per_replicate"][0]
    row["hybrid_route_receipt"]["routed_episodes"] = 1
    row["hybrid_route_receipt"]["route_fraction"] = 0.125
    row["hybrid_route_receipt"]["branch_route_mask"][0] = True

    with pytest.raises(ValueError, match="zero hybrid routing"):
        build_exp279_stop_path_ablation(payload)


def test_stop_path_ablation_rejects_confirmatory_boundary_crossing() -> None:
    payload = _payload()
    payload["confirmatory_data_consumed"] = True

    with pytest.raises(ValueError, match="confirmatory"):
        build_exp279_stop_path_ablation(payload)


def test_stop_path_sweep_preserves_lineage_and_orders_training_counts() -> None:
    payloads = [
        _payload(train_replicates=120),
        _payload(train_replicates=15),
        _payload(train_replicates=60),
    ]
    sweep = build_exp279_stop_path_sweep(payloads)

    assert sweep["schema"] == "NLM-EXP-279-STOP-PATH-SWEEP-DEV-V1"
    assert sweep["evidence_level"] == "EV-E2"
    assert sweep["decision"] == "UNVERIFIED"
    assert sweep["scientific_evidence_eligible"] is False
    assert sweep["training_replicate_counts"] == [15, 60, 120]
    assert sweep["evaluation_lineage"] == {"start_replicate": 20_000, "replicates": 2}
    assert sweep["all_hybrid_routes_zero"] is True
    assert sweep["challenge_materialized"] is False
    assert sweep["confirmatory_data_consumed"] is False
    assert [item["training_replicates"] for item in sweep["analyses"]] == [15, 60, 120]


def test_stop_path_validator_rejects_tampered_reconstruction() -> None:
    receipt = build_exp279_stop_path_ablation(_payload())
    tampered = deepcopy(receipt)
    tampered["reconstructed_utility_factor"] = 99.0

    errors = validate_exp279_stop_path_ablation(tampered)
    assert any("reconstruction" in error.lower() for error in errors)
