from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
RUN_KWARGS = dict(
    root_seed="exp279-paired-test",
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
    route_threshold=0.5,
    train_replicates=3,
    eval_replicates=6,
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


def _run():
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development

    return run_exp279_paired_development(**RUN_KWARGS)


def test_exp279_paired_runner_closes_development_lineage_without_promotion() -> None:
    from nolane_ai.experiments.exp279_paired_runner import validate_exp279_paired_development

    artifact = _run()
    assert artifact["schema"] == "NLM-EXP-279-PAIRED-DEV-EVAL-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["confirmatory_ready"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["decision_rule_executed"] is False
    assert artifact["protocol_id"] == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
    assert artifact["arm_order"] == ["propagation_only", "branch_only", "hybrid"]

    initial = artifact["initial_state"]
    assert initial["functional_digest_match"] is True
    assert len({initial["propagation_only_digest"], initial["branch_only_digest"], initial["hybrid_digest"]}) == 1

    resource = artifact["resource_match"]
    assert resource["parameter_match"] is True
    assert resource["functional_parameter_match"] is True
    assert resource["active_functional_parameter_match"] is True
    assert resource["reclaimed_parameter_assignment_closed"] is True
    assert resource["same_world_lineage"] is True
    assert resource["compute_budget_closed"] is True
    assert resource["structure_fit_strata"] == STRATA

    assert artifact["information_receipt"] == {
        "artifact": "constraint_variable_incidence",
        "ground_truth": True,
        "delivered_to": ["propagation_only", "hybrid"],
        "withheld_from": ["branch_only"],
        "branch_only_received_incidence": False,
    }
    assert artifact["route_config"] == {
        "strategy": "propagation_then_branch_on_residual_uncertainty",
        "threshold": 0.5,
        "frozen_before_evaluation": True,
    }
    assert artifact["primary_endpoint"] == {
        "metric": "verified_utility_per_accounted_flop_on_structure_dense_stratum",
        "direction": "higher",
        "mesi_relative_gain": 0.08,
    }
    assert artifact["protected_endpoints"]["verified_solution_rate_floor"] == "hybrid >= best_simple - 0.01"

    training = artifact["training"]
    assert training["rng_stream"] == "augmentation"
    assert training["start_replicate"] == 0
    assert training["replicates"] == 3
    assert training["strata"] == STRATA
    assert len(set(training["paired_batch_digests"])) == 3

    evaluation = artifact["evaluation"]
    assert evaluation["rng_stream"] == "evaluation"
    assert evaluation["start_replicate"] == 100
    assert evaluation["replicates"] == 6
    rows = evaluation["per_replicate"]
    assert [row["replicate"] for row in rows] == [100, 101, 102, 103, 104, 105]
    assert [row["stratum"] for row in rows] == STRATA * 2
    assert len({row["paired_batch_digest"] for row in rows}) == 6

    for row in rows:
        assert row["world_pairing_closed"] is True
        receipt = row["hybrid_route_receipt"]
        assert receipt["threshold"] == 0.5
        assert receipt["batch_size"] == 2
        assert receipt["routed_episodes"] == sum(receipt["branch_route_mask"])
        assert receipt["route_fraction"] == pytest.approx(receipt["routed_episodes"] / 2)
        for arm in ("propagation_only", "branch_only", "hybrid"):
            metrics = row[arm]
            assert metrics["accounted_flops_per_episode"] > 0
            assert metrics["verified_utility_per_accounted_flop_on_structure_dense_stratum"] >= 0.0

    aggregate = evaluation["aggregate"]
    assert aggregate["n"] == 6
    assert aggregate["structure_fit_strata"] == STRATA
    assert aggregate["best_simple_arm"] in {"propagation_only", "branch_only"}
    assert set(aggregate["by_stratum"]) == set(STRATA)
    assert validate_exp279_paired_development(artifact) == []


def test_exp279_runner_requires_all_predeclared_strata_and_disjoint_lineage() -> None:
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development

    with pytest.raises(ValueError, match="at least 3"):
        run_exp279_paired_development(**(RUN_KWARGS | {"eval_replicates": 2}))
    with pytest.raises(ValueError, match="disjoint"):
        run_exp279_paired_development(**(RUN_KWARGS | {"eval_start_replicate": 2}))


def test_exp279_validator_rejects_semantic_tamper_even_after_rehash() -> None:
    from nolane_ai.experiments.exp279_paired_runner import _artifact_digest, validate_exp279_paired_development

    original = _run()
    mutations = []

    bad = deepcopy(original)
    bad["primary_endpoint"]["metric"] = "accuracy"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["primary_endpoint"]["mesi_relative_gain"] = 0.01
    mutations.append(bad)

    bad = deepcopy(original)
    bad["protected_endpoints"]["verified_solution_rate_floor"] = "hybrid >= best_simple - 0.10"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["arm_order"] = list(reversed(bad["arm_order"]))
    mutations.append(bad)

    bad = deepcopy(original)
    bad["information_receipt"]["branch_only_received_incidence"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["stratum"] = "BRANCH_FIT"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["route_config"]["threshold"] = 0.25
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["hybrid_route_receipt"]["routed_episodes"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["resource_match"]["compute_budget_closed"] = False
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["aggregate"]["mean_hybrid_utility"] += 0.1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["confirmatory_data_consumed"] = True
    mutations.append(bad)

    for payload in mutations:
        payload["artifact_digest"] = _artifact_digest(payload)
        assert validate_exp279_paired_development(payload)
