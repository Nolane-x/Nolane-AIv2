from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


RUN_KWARGS = dict(
    root_seed="exp289-paired-test",
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
    train_replicates=2,
    eval_replicates=3,
    eval_start_replicate=100,
    batch_size=2,
    timesteps=3,
    restarts=3,
    variables=6,
    decoys=2,
    max_search_steps=10,
    noise_std=0.05,
    lr=1e-3,
    weight_decay=0.0,
    protocol_digest="p" * 64,
    code_digest="c" * 64,
)


def _run(**overrides):
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development

    return run_exp289_paired_development(**(RUN_KWARGS | overrides))


def test_exp289_paired_runner_closes_development_boundary_lineage_and_scope() -> None:
    from nolane_ai.experiments.exp289_paired_runner import (
        validate_exp289_execution_artifact,
        validate_exp289_paired_development,
    )

    artifact = _run()
    assert artifact["schema"] == "NLM-EXP-289-PAIRED-DEV-EVAL-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["confirmatory_ready"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_seed_materialized"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["decision_rule_executed"] is False
    assert artifact["protocol_id"] == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
    assert artifact["experiment_id"] == "EXP-289"
    assert artifact["arm_order"] == ["no_nogood", "local_nogood"]
    assert artifact["primary_endpoint"] == {
        "metric": "repeat_dead_end_rate",
        "direction": "lower",
        "mesi_relative_reduction": 0.25,
    }
    assert artifact["protected_endpoints"] == {
        "valid_state_overprune_rate_ceiling": 0.005,
        "verified_solution_rate_floor": "local_nogood >= no_nogood - 0.01",
    }
    assert artifact["multiplicity_family"] == "LOCAL_NOGOOD"
    assert artifact["scope_policy"] == {
        "scope": "episode_local",
        "cross_episode_reuse": False,
        "cross_problem_reuse": False,
        "exact_subset_matching_only": True,
        "oracle_conflict_core_used": False,
        "ground_truth_gates_arm_action": False,
        "learned_clause_generalization_claimed": False,
    }

    initial = artifact["initial_state"]
    assert initial["functional_digest_match"] is True
    assert initial["no_nogood_digest"] == initial["local_nogood_digest"]
    assert artifact["model_init_rng_stream"] == "model_init"

    resource = artifact["resource_match"]
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "same_initial_world_lineage",
        "memory_scope_closed",
        "compute_budget_closed",
    ):
        assert resource[key] is True
    assert resource["pair_audit"]["schema"] == "NLM-EXP-289-MATCHED-NOGOOD-ARMS-DEV-V1"
    assert resource["declared_max_accounted_cost_per_episode"] > 0
    assert resource["hardware_profiler_flops_claimed"] is False

    training = artifact["training"]
    assert training["rng_stream"] == "augmentation"
    assert training["start_replicate"] == 0
    assert training["replicates"] == 2
    assert len(training["paired_batch_digests"]) == 2
    assert len(set(training["paired_batch_digests"])) == 2

    evaluation = artifact["evaluation"]
    assert evaluation["rng_stream"] == "evaluation"
    assert evaluation["start_replicate"] == 100
    assert evaluation["replicates"] == 3
    rows = evaluation["per_replicate"]
    assert [row["replicate"] for row in rows] == [100, 101, 102]
    assert len({row["paired_batch_digest"] for row in rows}) == 3

    ceiling = resource["declared_max_accounted_cost_per_episode"]
    for row in rows:
        assert row["initial_world_pairing_closed"] is True
        assert row["opportunity_manifest_pairing_closed"] is True
        assert row["evaluator_metadata_delivered_to_arm"] is False
        baseline = row["no_nogood"]
        local = row["local_nogood"]
        assert baseline["predeclared_repeat_opportunities"] == local[
            "predeclared_repeat_opportunities"
        ]
        assert baseline["predeclared_repeat_opportunities"] > 0

        for arm in (baseline, local):
            denominator = arm["predeclared_repeat_opportunities"]
            numerator = arm["repeated_dead_end_reentries"]
            assert 0 <= numerator <= denominator
            assert arm["repeat_dead_end_rate"] == pytest.approx(numerator / denominator)
            assert 0.0 <= arm["verified_solution_rate"] <= 1.0
            assert 0.0 <= arm["valid_state_overprune_rate"] <= 1.0
            assert 0 < arm["accounted_reasoning_cost"] <= ceiling
            assert len(arm["episodes"]) == RUN_KWARGS["batch_size"]
            if arm["censored_at_max_cost"]:
                assert arm["accounted_reasoning_cost"] == ceiling

        for episode in local["episodes"]:
            assert episode["scope_receipt"]["episode_digest"] == episode["episode_digest"]
            assert episode["scope_receipt"]["problem_digest"] == episode["problem_digest"]
            assert episode["scope_receipt"]["cross_episode_reuse"] is False
            assert episode["scope_receipt"]["cross_problem_reuse"] is False
            assert episode["evaluator_metadata_delivered_to_arm"] is False
            assert episode["predeclared_repeat_opportunities"] > 0
            assert episode["rder_defined"] is True
            assert episode["repeat_dead_end_rate"] == pytest.approx(
                episode["repeated_dead_end_reentries"]
                / episode["predeclared_repeat_opportunities"]
            )
            for insertion in episode["store_insertion_receipts"]:
                assert insertion["partial_assignment_reached"] is True
                assert insertion["dead_end_observed_before_insertion"] is True
                assert insertion["canonical_key"]
                assert insertion["canonical_key"] == sorted(insertion["canonical_key"])
                assert insertion["episode_digest"] == episode["episode_digest"]
                assert insertion["problem_digest"] == episode["problem_digest"]
                assert insertion["oracle_conflict_core_used"] is False
                assert insertion["ground_truth_safety_gate_used"] is False
                assert insertion["future_path_used"] is False
                assert insertion["evaluator_result_delivered_to_arm"] is False
                assert isinstance(insertion["posthoc_has_valid_completion"], bool)
                assert insertion["charged_canonicalization_operations"] > 0
                assert insertion["charged_insertion_operations"] in {0, 1}

    aggregate = evaluation["aggregate"]
    assert aggregate["n"] == 3
    assert aggregate["analysis_boundary"] == (
        "descriptive DEVELOPMENT statistics only; frozen confirmatory bootstrap inference not executed"
    )
    assert aggregate["zero_opportunity_episode_count"] == 0
    assert validate_exp289_paired_development(artifact) == []
    assert validate_exp289_execution_artifact(artifact) is None


def test_exp289_local_store_changes_only_causal_memory_path_and_charges_real_work() -> None:
    artifact = _run()
    for row in artifact["evaluation"]["per_replicate"]:
        baseline = row["no_nogood"]
        local = row["local_nogood"]
        assert baseline["memory_insertion_count"] == 0
        assert baseline["nogood_hits"] == 0
        assert baseline["prevented_repeat_count"] == 0
        assert local["memory_insertion_count"] > 0
        assert local["memory_query_count"] > 0
        assert local["memory_canonicalization_operations"] > 0
        assert local["memory_comparison_count"] > 0
        assert local["nogood_hits"] >= local["prevented_repeat_count"]
        assert local["repeated_dead_end_reentries"] <= baseline["repeated_dead_end_reentries"]
        assert local["predeclared_repeat_opportunities"] == baseline[
            "predeclared_repeat_opportunities"
        ]
        assert row["search_path_pairing_policy"] == (
            "same_initial_world_restart_lineage_with_causal_post_memory_divergence_preserved"
        )


def test_exp289_zero_opportunity_episodes_are_retained_and_excluded_only_from_rder() -> None:
    artifact = _run(decoys=0)
    rows = artifact["evaluation"]["per_replicate"]
    assert len(rows) == RUN_KWARGS["eval_replicates"]
    for row in rows:
        for arm_name in ("no_nogood", "local_nogood"):
            arm = row[arm_name]
            assert arm["predeclared_repeat_opportunities"] == 0
            assert arm["repeat_dead_end_rate"] is None
            for episode in arm["episodes"]:
                assert episode["predeclared_repeat_opportunities"] == 0
                assert episode["rder_defined"] is False
                assert episode["repeat_dead_end_rate"] is None
                assert episode["zero_opportunity_policy"] == (
                    "retain_raw_episode_exclude_from_rder_denominator"
                )
    assert artifact["evaluation"]["aggregate"]["zero_opportunity_episode_count"] > 0


def test_exp289_runner_requires_disjoint_development_lineage() -> None:
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development

    with pytest.raises(ValueError, match="disjoint"):
        run_exp289_paired_development(**(RUN_KWARGS | {"eval_start_replicate": 1}))


def test_exp289_validator_rejects_rehashed_semantic_tamper() -> None:
    from nolane_ai.experiments.exp289_paired_runner import (
        _artifact_digest,
        validate_exp289_paired_development,
    )

    original = _run()
    mutations = []

    bad = deepcopy(original)
    bad["primary_endpoint"]["metric"] = "accuracy"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["primary_endpoint"]["mesi_relative_reduction"] = 0.01
    mutations.append(bad)

    bad = deepcopy(original)
    bad["protected_endpoints"]["valid_state_overprune_rate_ceiling"] = 0.5
    mutations.append(bad)

    bad = deepcopy(original)
    bad["scope_policy"]["cross_episode_reuse"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["resource_match"]["optimizer_visible_parameter_match"] = False
    mutations.append(bad)

    bad = deepcopy(original)
    bad["training"]["replicates"] = 101
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"] = list(reversed(bad["evaluation"]["per_replicate"]))
    mutations.append(bad)

    bad = deepcopy(original)
    arm = bad["evaluation"]["per_replicate"][0]["local_nogood"]
    arm["predeclared_repeat_opportunities"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    arm = bad["evaluation"]["per_replicate"][0]["local_nogood"]
    arm["repeated_dead_end_reentries"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    insertion = bad["evaluation"]["per_replicate"][0]["local_nogood"]["episodes"][0][
        "store_insertion_receipts"
    ][0]
    solution_pair = bad["evaluation"]["per_replicate"][0]["world_receipts"][0][
        "valid_solutions"
    ][0][0]
    insertion["canonical_key"] = [solution_pair]
    insertion["posthoc_has_valid_completion"] = False
    mutations.append(bad)

    bad = deepcopy(original)
    episode = bad["evaluation"]["per_replicate"][0]["local_nogood"]["episodes"][0]
    episode["accounted_reasoning_cost"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["aggregate"]["mean_local_repeat_dead_end_rate"] += 0.1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["confirmatory_data_consumed"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["challenge_seed_materialized"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["decision_rule_executed"] = True
    mutations.append(bad)

    for payload in mutations:
        payload["artifact_digest"] = _artifact_digest(payload)
        assert validate_exp289_paired_development(payload)
