from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


RUN_KWARGS = dict(
    root_seed="exp286-paired-test",
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
    train_replicates=2,
    eval_replicates=3,
    eval_start_replicate=100,
    batch_size=2,
    timesteps=3,
    variables=5,
    decoys=2,
    max_search_steps=8,
    noise_std=0.05,
    lr=1e-3,
    weight_decay=0.0,
    protocol_digest="p" * 64,
    code_digest="c" * 64,
)


def _run(**overrides):
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development

    return run_exp286_paired_development(**(RUN_KWARGS | overrides))


def test_exp286_paired_runner_closes_development_boundary_and_lineage() -> None:
    from nolane_ai.experiments.exp286_paired_runner import validate_exp286_paired_development

    artifact = _run()
    assert artifact["schema"] == "NLM-EXP-286-PAIRED-DEV-EVAL-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["confirmatory_ready"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_seed_materialized"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["decision_rule_executed"] is False
    assert artifact["protocol_id"] == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
    assert artifact["experiment_id"] == "EXP-286"
    assert artifact["arm_order"] == ["chronological_failure", "oracle_conflict_core"]
    assert artifact["primary_endpoint"] == {
        "metric": "accounted_reasoning_flops_to_verified_solution",
        "direction": "lower",
        "mesi_relative_reduction": 0.15,
    }
    assert artifact["protected_endpoints"]["verified_solution_rate_floor"] == (
        "oracle_conflict_core >= chronological_failure - 0.005"
    )
    assert artifact["multiplicity_family"] == "CONFLICT_VALUE"
    assert artifact["information_receipt"] == {
        "artifact": "ground_truth_conflict_core",
        "ground_truth": True,
        "delivery_event": "after_current_contradiction_only",
        "delivered_to": ["oracle_conflict_core"],
        "withheld_from": ["chronological_failure"],
        "chronological_failure_received_conflict_core": False,
        "future_conflict_core_leakage": False,
        "solution_leakage": False,
    }

    initial = artifact["initial_state"]
    assert initial["functional_digest_match"] is True
    assert initial["chronological_failure_digest"] == initial["oracle_conflict_core_digest"]

    resource = artifact["resource_match"]
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "same_initial_world_lineage",
        "compute_budget_closed",
        "oracle_information_separation",
    ):
        assert resource[key] is True
    assert resource["pair_audit"]["schema"] == "NLM-EXP-286-MATCHED-CONFLICT-ARMS-DEV-V1"
    assert resource["declared_max_accounted_flops_per_episode"] > 0

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

    ceiling = resource["declared_max_accounted_flops_per_episode"]
    for row in rows:
        assert row["initial_world_pairing_closed"] is True
        assert row["future_conflict_core_leakage"] is False
        assert row["solution_leakage"] is False
        for arm in ("chronological_failure", "oracle_conflict_core"):
            metrics = row[arm]
            assert 0 < metrics["accounted_reasoning_flops_to_verified_solution"] <= ceiling
            assert metrics["verified_solution_rate"] in {0.0, 1.0}
            assert metrics["search_steps"] >= 1
            assert isinstance(metrics["visited_variables"], list)
            assert metrics["contradiction_count"] >= 0
            if metrics["verified_solution_rate"] == 0.0:
                assert metrics["censored_at_max_flops"] is True
                assert metrics["accounted_reasoning_flops_to_verified_solution"] == ceiling
            else:
                assert metrics["censored_at_max_flops"] is False
                assert metrics["external_solution_verified"] is True
                assert metrics["verified_solution_digest"]

        assert row["chronological_failure"]["conflict_core_received"] is False
        oracle = row["oracle_conflict_core"]
        assert oracle["conflict_core_precontradiction_delivery"] is False
        assert oracle["conflict_core_delivery_count"] <= oracle["contradiction_count"]

    aggregate = evaluation["aggregate"]
    assert aggregate["n"] == 3
    assert aggregate["analysis_boundary"] == (
        "descriptive DEVELOPMENT statistics only; frozen confirmatory bootstrap inference not executed"
    )
    assert validate_exp286_paired_development(artifact) == []


def test_exp286_runner_retains_unresolved_episodes_as_censored_outcomes() -> None:
    artifact = _run(max_search_steps=1)
    ceiling = artifact["resource_match"]["declared_max_accounted_flops_per_episode"]
    rows = artifact["evaluation"]["per_replicate"]
    censored = []
    for row in rows:
        for arm in ("chronological_failure", "oracle_conflict_core"):
            metrics = row[arm]
            if metrics["verified_solution_rate"] == 0.0:
                censored.append(metrics)
                assert metrics["censored_at_max_flops"] is True
                assert metrics["accounted_reasoning_flops_to_verified_solution"] == ceiling
    assert censored, "a one-step search budget must retain at least one unresolved censored episode"


def test_exp286_runner_requires_disjoint_development_lineage() -> None:
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development

    with pytest.raises(ValueError, match="disjoint"):
        run_exp286_paired_development(**(RUN_KWARGS | {"eval_start_replicate": 1}))


def test_exp286_validator_rejects_rehashed_semantic_tamper() -> None:
    from nolane_ai.experiments.exp286_paired_runner import _artifact_digest, validate_exp286_paired_development

    original = _run()
    mutations = []

    bad = deepcopy(original)
    bad["primary_endpoint"]["metric"] = "accuracy"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["primary_endpoint"]["mesi_relative_reduction"] = 0.01
    mutations.append(bad)

    bad = deepcopy(original)
    bad["protected_endpoints"]["verified_solution_rate_floor"] = (
        "oracle_conflict_core >= chronological_failure - 0.50"
    )
    mutations.append(bad)

    bad = deepcopy(original)
    bad["arm_order"] = list(reversed(bad["arm_order"]))
    mutations.append(bad)

    bad = deepcopy(original)
    bad["information_receipt"]["chronological_failure_received_conflict_core"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["chronological_failure"]["conflict_core_received"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["oracle_conflict_core"][
        "conflict_core_precontradiction_delivery"
    ] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["future_conflict_core_leakage"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["resource_match"]["optimizer_visible_parameter_match"] = False
    mutations.append(bad)

    bad = deepcopy(original)
    bad["resource_match"]["declared_max_accounted_flops_per_episode"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    solved = None
    for row in bad["evaluation"]["per_replicate"]:
        for arm in ("chronological_failure", "oracle_conflict_core"):
            if row[arm]["verified_solution_rate"] == 1.0:
                solved = row[arm]
                break
        if solved is not None:
            break
    if solved is not None:
        solved["external_solution_verified"] = False
        mutations.append(bad)

    bad = deepcopy(original)
    metrics = bad["evaluation"]["per_replicate"][0]["chronological_failure"]
    metrics["verified_solution_rate"] = 0.0
    metrics["censored_at_max_flops"] = False
    metrics["accounted_reasoning_flops_to_verified_solution"] = 1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"] = list(reversed(bad["evaluation"]["per_replicate"]))
    mutations.append(bad)

    bad = deepcopy(original)
    bad["training"]["replicates"] = 101
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["aggregate"]["mean_chronological_cost"] += 1.0
    mutations.append(bad)

    bad = deepcopy(original)
    bad["confirmatory_data_consumed"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["challenge_seed_materialized"] = True
    mutations.append(bad)

    for payload in mutations:
        payload["artifact_digest"] = _artifact_digest(payload)
        assert validate_exp286_paired_development(payload)


def test_exp286_validator_recomputes_critical_resource_and_episode_receipts_after_rehash() -> None:
    from nolane_ai.experiments.exp286_paired_runner import _artifact_digest, validate_exp286_paired_development
    from nolane_ai.protocol.evidence import canonical_sha256

    original = _run()
    mutations = []

    bad = deepcopy(original)
    bad["initial_state"]["chronological_failure_digest"] = "f" * 64
    bad["initial_state"]["oracle_conflict_core_digest"] = "f" * 64
    mutations.append(bad)

    bad = deepcopy(original)
    bad["model_init_seed"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    bad_pair = bad["resource_match"]["pair_audit"]
    bad_pair["chronological_failure"]["total_parameters"] += 1
    bad["resource_match"]["pair_audit_digest"] = canonical_sha256(bad_pair)
    mutations.append(bad)

    bad = deepcopy(original)
    episode = bad["evaluation"]["per_replicate"][0]["chronological_failure"]["episodes"][0]
    episode["actual_executed_flops_before_censoring"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    episode = bad["evaluation"]["per_replicate"][0]["chronological_failure"]["episodes"][0]
    episode["search_steps"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    episode = bad["evaluation"]["per_replicate"][0]["chronological_failure"]["episodes"][0]
    episode["contradiction_count"] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    episode = bad["evaluation"]["per_replicate"][0]["chronological_failure"]["episodes"][0]
    episode["candidate_solution"] = []
    mutations.append(bad)

    bad = deepcopy(original)
    receipt = bad["evaluation"]["per_replicate"][0]["chronological_failure"]["episodes"][0][
        "step_receipts"
    ][0]
    receipt["external_target_value"] = 1 - int(receipt["external_target_value"])
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"][0]["chronological_failure"][
        "censored_episode_count"
    ] += 1
    mutations.append(bad)

    bad = deepcopy(original)
    bad["analysis_method_boundary"] = "confirmatory inference executed"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["learned_conflict_localizer_validated"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["remaining_blockers"] = []
    mutations.append(bad)

    for payload in mutations:
        payload["artifact_digest"] = _artifact_digest(payload)
        assert validate_exp286_paired_development(payload)
