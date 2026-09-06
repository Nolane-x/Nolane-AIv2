from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


RUN_KWARGS = dict(
    root_seed="exp277-paired-test",
    d_model=8,
    hidden_size=6,
    target_parameters=5_000,
    train_replicates=2,
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


def _run():
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development

    return run_exp277_paired_development(**RUN_KWARGS)


def test_exp277_paired_runner_closes_development_lineage_without_promotion() -> None:
    from nolane_ai.experiments.exp277_paired_runner import validate_exp277_paired_development

    artifact = _run()
    assert artifact["schema"] == "NLM-EXP-277-PAIRED-DEV-EVAL-V1"
    assert artifact["evidence_level"] == "EV-E2"
    assert artifact["decision"] == "UNVERIFIED"
    assert artifact["confirmatory_ready"] is False
    assert artifact["confirmatory_data_consumed"] is False
    assert artifact["challenge_materialized"] is False
    assert artifact["initial_state"]["functional_digest_match"] is True
    assert artifact["resource_match"]["parameter_match"] is True
    assert artifact["resource_match"]["functional_parameter_match"] is True
    assert artifact["resource_match"]["same_world_lineage"] is True
    assert artifact["resource_match"]["compute_budget_closed"] is True
    assert artifact["oracle_information_receipt"] == {
        "artifact": "oracle_incidence",
        "ground_truth": True,
        "delivered_to": ["oracle_cbrf"],
        "withheld_from": ["arcs_branch"],
        "arcs_received_oracle_incidence": False,
    }
    assert artifact["training"]["rng_stream"] == "augmentation"
    assert artifact["training"]["start_replicate"] == 0
    assert artifact["training"]["replicates"] == 2
    assert artifact["evaluation"]["rng_stream"] == "evaluation"
    assert artifact["evaluation"]["start_replicate"] == 100
    assert [row["replicate"] for row in artifact["evaluation"]["per_replicate"]] == [100, 101, 102]
    assert len({row["paired_batch_digest"] for row in artifact["evaluation"]["per_replicate"]}) == 3
    assert artifact["primary_endpoint"] == {
        "metric": "verified_utility_per_accounted_flop",
        "direction": "higher",
        "mesi_relative_gain": 0.10,
    }
    assert artifact["protected_endpoints"]["verified_solution_rate_floor"] == "oracle_cbrf >= arcs_branch - 0.005"
    assert artifact["decision_rule_executed"] is False
    assert validate_exp277_paired_development(artifact) == []


def test_exp277_runner_rejects_training_evaluation_replicate_overlap() -> None:
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development

    with pytest.raises(ValueError, match="disjoint"):
        run_exp277_paired_development(**(RUN_KWARGS | {"eval_start_replicate": 1}))


def test_exp277_validator_rejects_semantic_tamper_even_after_rehash() -> None:
    from nolane_ai.experiments.exp277_paired_runner import _artifact_digest, validate_exp277_paired_development

    original = _run()
    mutations = []

    bad = deepcopy(original)
    bad["primary_endpoint"]["metric"] = "accuracy"
    mutations.append(bad)

    bad = deepcopy(original)
    bad["primary_endpoint"]["mesi_relative_gain"] = 0.01
    mutations.append(bad)

    bad = deepcopy(original)
    bad["oracle_information_receipt"]["arcs_received_oracle_incidence"] = True
    mutations.append(bad)

    bad = deepcopy(original)
    bad["resource_match"]["same_world_lineage"] = False
    mutations.append(bad)

    bad = deepcopy(original)
    bad["evaluation"]["per_replicate"] = list(reversed(bad["evaluation"]["per_replicate"]))
    mutations.append(bad)

    for payload in mutations:
        payload["artifact_digest"] = _artifact_digest(payload)
        assert validate_exp277_paired_development(payload)
