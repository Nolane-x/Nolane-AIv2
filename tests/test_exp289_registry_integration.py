from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


def _protocol():
    return {
        "protocol_id": "NLM-REASONING-STAGE-A-CONFIRMATORY-V1",
        "status": "FROZEN_V1",
        "experiments": [
            {
                "experiment_id": "EXP-277",
                "arms": [
                    {"id": "arcs_branch", "description": "V0.15 ARCS recurrent-depth + branch bank + verifier court"},
                    {"id": "oracle_cbrf", "description": "same substrate with ground-truth constraint/factor representation"},
                ],
                "resource_match": {"parameter_budget": "matched active parameter count", "inference_budget": "same max accounted FLOPs per episode", "world_pairing": "same world lineage and replicate index"},
            },
            {
                "experiment_id": "EXP-279",
                "arms": [
                    {"id": "propagation_only", "description": "constraint propagation without branch search"},
                    {"id": "branch_only", "description": "ARCS branch search without CBRF propagation"},
                    {"id": "hybrid", "description": "propagation followed by branch search when residual uncertainty remains"},
                ],
                "resource_match": {"parameter_budget": "reclaimed parameters assigned to simpler rivals", "inference_budget": "equal max accounted FLOPs", "structure_fit": "predeclared strata"},
            },
            {
                "experiment_id": "EXP-282",
                "arms": [
                    {"id": "recurrent_hidden", "description": "matched recurrent state without explicit belief representation"},
                    {"id": "explicit_belief", "description": "explicit calibrated belief state over hidden world variables"},
                ],
                "resource_match": {"parameter_budget": "equal state/controller parameters", "observation_history": "identical", "compute_budget": "matched"},
            },
            {
                "experiment_id": "EXP-286",
                "arms": [
                    {"id": "chronological_failure", "description": "no conflict core; chronological rollback"},
                    {"id": "oracle_conflict_core", "description": "ground-truth conflict core supplied at contradiction"},
                ],
                "resource_match": {"parameter_budget": "oracle metadata is information, not free neural parameters; oracle-info receipt reported", "episode_budget": "same max FLOPs"},
            },
            {
                "experiment_id": "EXP-289",
                "arms": [
                    {"id": "no_nogood", "description": "hybrid reasoner without episode-local learned nogoods"},
                    {"id": "local_nogood", "description": "same reasoner with episode-local nogood store and scoped applicability"},
                ],
                "resource_match": {"parameter_budget": "matched", "memory_cost": "nogood storage/retrieval charged to accounted cost", "episode_budget": "matched"},
            },
        ],
    }


def _model_audit():
    return audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))


def _execution():
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development

    return run_exp289_paired_development(
        root_seed="exp289-registry-test",
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


def _raw_summary(execution):
    rows = execution["evaluation"]["per_replicate"]
    world_receipts = [receipt for row in rows for receipt in row["world_receipts"]]
    local_episodes = [episode for row in rows for episode in row["local_nogood"]["episodes"]]
    baseline_episodes = [episode for row in rows for episode in row["no_nogood"]["episodes"]]
    return {
        "rder_denominator_eligible_episode_count": sum(
            int(receipt["predeclared_repeat_opportunities"]) > 0 for receipt in world_receipts
        ),
        "predeclared_repeat_opportunity_count": sum(
            int(receipt["predeclared_repeat_opportunities"]) for receipt in world_receipts
        ),
        "no_nogood_repeated_dead_end_reentries": sum(
            int(episode["repeated_dead_end_reentries"]) for episode in baseline_episodes
        ),
        "local_nogood_repeated_dead_end_reentries": sum(
            int(episode["repeated_dead_end_reentries"]) for episode in local_episodes
        ),
        "local_prevented_repeat_count": sum(
            int(episode["prevented_repeat_count"]) for episode in local_episodes
        ),
        "local_nogood_hit_count": sum(int(episode["nogood_hits"]) for episode in local_episodes),
        "local_nogood_store_count": sum(
            int(episode["memory_insertion_count"]) for episode in local_episodes
        ),
        "local_overprune_event_count": sum(
            int(episode["invalid_valid_state_prune_count"]) for episode in local_episodes
        ),
        "censored_episode_count": sum(
            int(bool(episode["censored_at_max_cost"]))
            for episode in baseline_episodes + local_episodes
        ),
    }


def test_registry_accepts_exp289_pair_and_execution_but_keeps_match_court_blocked() -> None:
    from nolane_ai.experiments.neural_arm_registry import (
        build_neural_arm_registry,
        validate_neural_arm_registry,
    )

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp289_pair_audit=pair_audit,
        exp289_execution_artifact=execution,
    )
    entry = registry["experiments"]["EXP-289"]
    assert entry["development_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
    assert entry["development_match_status"] == "PAIRED_LOCAL_NOGOOD_DEV_READY"
    assert entry["match_court"] == "BLOCKED"
    assert entry["evidence_level"] == "EV-E2"
    assert entry["decision"] == "UNVERIFIED"
    assert entry["arms"]["no_nogood"]["implementation_status"] == "IMPLEMENTED"
    assert entry["arms"]["local_nogood"]["implementation_status"] == "IMPLEMENTED"

    evidence = entry["resource_match_evidence"]
    for key in (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "memory_scope_closed",
        "compute_budget_closed",
    ):
        assert evidence[key] is True
    assert evidence["scope"] == "episode_local"
    assert evidence["cross_episode_reuse"] is False
    assert evidence["cross_problem_reuse"] is False
    assert evidence["exact_subset_matching_only"] is True
    assert evidence["ground_truth_gates_arm_action"] is False

    paired = entry["paired_execution_evidence"]
    assert paired["artifact_digest"] == execution["artifact_digest"]
    assert paired["code_digest"] == execution["code_digest"]
    assert paired["training_replicates"] == 2
    assert paired["evaluation_replicates"] == 3
    for key, value in _raw_summary(execution).items():
        assert paired[key] == value
    assert paired["accounted_cost_ceiling"] == execution["resource_match"][
        "declared_max_accounted_cost_per_episode"
    ]
    assert paired["learned_clause_transfer_validated"] is False
    assert paired["lifelong_lemma_economy_validated"] is False
    assert validate_neural_arm_registry(registry) == []


def test_registry_accepts_exp289_pair_only_without_claiming_execution_readiness() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp289_pair_audit=execution["resource_match"]["pair_audit"],
    )
    entry = registry["experiments"]["EXP-289"]
    assert entry["development_status"] == "PARAMETER_COMPUTE_SCOPE_COST_CLOSED"
    assert entry["development_match_status"] == "PARAMETER_COMPUTE_SCOPE_COST_CLOSED"
    assert entry["match_court"] == "BLOCKED"
    assert "paired_execution_evidence" not in entry
    assert entry["blockers"] == [
        "EXP-289: matched arms are not yet integrated into paired episode-local nogood DEVELOPMENT execution lineage"
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("parameter_match",), False),
        (("compute_budget_closed",), False),
        (("memory_scope_closed",), False),
        (("memory_scope_receipt", "cross_episode_reuse"), True),
        (("memory_scope_receipt", "cross_problem_reuse"), True),
        (("memory_scope_receipt", "exact_subset_matching_only"), False),
        (("memory_scope_receipt", "ground_truth_gates_arm_action"), True),
    ],
)
def test_registry_rejects_exp289_pair_scope_parameter_or_cost_drift(path, value) -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    pair_audit = deepcopy(_execution()["resource_match"]["pair_audit"])
    target = pair_audit
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="EXP-289 matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp289_pair_audit=pair_audit,
        )


def test_registry_rejects_exp289_execution_without_pair_audit() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    with pytest.raises(ValueError, match="requires matched pair audit"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp289_execution_artifact=_execution(),
        )


def test_registry_rejects_exp289_execution_protocol_or_scope_tamper_after_rehash() -> None:
    from nolane_ai.experiments.exp289_paired_runner import _artifact_digest
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]

    bad_protocol = deepcopy(execution)
    bad_protocol["protocol_digest"] = "q" * 64
    bad_protocol["artifact_digest"] = _artifact_digest(bad_protocol)
    with pytest.raises(ValueError, match="protocol digest mismatch|EXP-289 paired execution artifact"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp289_pair_audit=pair_audit,
            exp289_execution_artifact=bad_protocol,
        )

    bad_scope = deepcopy(execution)
    bad_scope["scope_policy"]["cross_episode_reuse"] = True
    bad_scope["artifact_digest"] = _artifact_digest(bad_scope)
    with pytest.raises(ValueError, match="EXP-289 paired execution artifact"):
        build_neural_arm_registry(
            protocol=_protocol(),
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
            exp289_pair_audit=pair_audit,
            exp289_execution_artifact=bad_scope,
        )


def test_registry_rejects_exp289_learned_transfer_lifelong_or_confirmatory_overclaim() -> None:
    from nolane_ai.experiments.exp289_paired_runner import _artifact_digest
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    execution = _execution()
    pair_audit = execution["resource_match"]["pair_audit"]
    mutations = (
        ("learned_clause_transfer_validated", True),
        ("lifelong_lemma_economy_validated", True),
        ("confirmatory_ready", True),
    )
    for field, value in mutations:
        overclaim = deepcopy(execution)
        overclaim[field] = value
        overclaim["artifact_digest"] = _artifact_digest(overclaim)
        with pytest.raises(ValueError, match="EXP-289 paired execution artifact|EXP-289 .*validated"):
            build_neural_arm_registry(
                protocol=_protocol(),
                protocol_digest="p" * 64,
                model_audit=_model_audit(),
                exp289_pair_audit=pair_audit,
                exp289_execution_artifact=overclaim,
            )


def test_registry_validator_rejects_rehashed_exp289_blocker_or_summary_tamper() -> None:
    from nolane_ai.experiments.neural_arm_registry import (
        _digest,
        build_neural_arm_registry,
        validate_neural_arm_registry,
    )

    execution = _execution()
    registry = build_neural_arm_registry(
        protocol=_protocol(),
        protocol_digest="p" * 64,
        model_audit=_model_audit(),
        exp289_pair_audit=execution["resource_match"]["pair_audit"],
        exp289_execution_artifact=execution,
    )

    no_blockers = deepcopy(registry)
    no_blockers["experiments"]["EXP-289"]["blockers"] = []
    no_blockers["registry_digest"] = _digest(no_blockers)
    assert validate_neural_arm_registry(no_blockers)

    forged = deepcopy(registry)
    forged["experiments"]["EXP-289"]["paired_execution_evidence"][
        "local_nogood_repeated_dead_end_reentries"
    ] += 1
    forged["registry_digest"] = _digest(forged)
    errors = validate_neural_arm_registry(forged)
    assert "EXP-289 paired execution evidence summary drift" in errors


def test_registry_rejects_exp289_protocol_arm_description_drift() -> None:
    from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry

    protocol = _protocol()
    protocol["experiments"][-1]["arms"][1]["description"] = "same reasoner with global learned clauses"
    with pytest.raises(ValueError, match="protocol arm drift for EXP-289"):
        build_neural_arm_registry(
            protocol=protocol,
            protocol_digest="p" * 64,
            model_audit=_model_audit(),
        )
