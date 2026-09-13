from __future__ import annotations

import inspect
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def _tiny_geometry() -> dict[str, object]:
    return {
        "batch_size": 2,
        "canonical_indices": [0, 1, 2, 3],
        "d_model": 64,
        "decoys": 3,
        "eval_replicates": 1,
        "eval_start_replicate": 40000,
        "hidden_size": 48,
        "lr": 0.002,
        "max_search_steps": 24,
        "noise_std": 0.05,
        "restarts": 4,
        "root_prefix": "20260913-exp290-structural-clause-transfer-v1-dev",
        "target_parameters": 500000,
        "timesteps": 4,
        "train_replicates": 1,
        "variables": 8,
        "weight_decay": 0.0,
    }


def _pair():
    from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator

    return Exp290TransferGenerator(root_seed="20260913-exp290-task2-test").make_pair(
        replicate=40000,
        batch_size=2,
        timesteps=4,
        restarts=4,
        variables=8,
        decoys=3,
        d_model=64,
        noise_std=0.05,
        rng_stream="evaluation",
        device="cpu",
    )


def test_exp290_common_source_phase_uses_only_reached_public_contradictions() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import build_common_source_phase

    pair = _pair()
    source = build_common_source_phase(pair, max_search_steps=24)

    assert source["source_clause_set_sealed_before_target"] is True
    assert source["evaluator_truth_gated_insertion"] is False
    assert source["oracle_mapping_used"] is False
    assert len(source["episodes"]) == 2
    assert any(row["source_clauses"] for row in source["episodes"])
    for episode in source["episodes"]:
        for row in episode["insertion_receipts"]:
            assert row["partial_assignment_reached"] is True
            assert row["public_contradiction_observed_before_insertion"] is True
            assert row["evaluator_truth_used_for_insertion"] is False
            assert row["future_target_truth_used"] is False
            assert len(row["clause"]) == 1


def test_exp290_training_contract_is_fixed_augmentation_only_111() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import training_contract_receipt

    receipt = training_contract_receipt()
    assert receipt == {
        "rng_stream": "augmentation",
        "evaluation_mapping_used_for_training": False,
        "evaluation_lineage_consumed": False,
        "loss_weights": {
            "branch_cross_entropy": 1.0,
            "verifier_binary_cross_entropy": 1.0,
            "transfer_mapping_cross_entropy": 1.0,
        },
        "class_weights_used": False,
        "calibration_used": False,
        "early_stopping": False,
        "hard_negative_mining": False,
    }


def test_exp290_tiny_root_keeps_parent_local_memory_in_every_target_mode() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import run_exp290_root

    result = run_exp290_root(
        canonical_index=0,
        geometry=_tiny_geometry(),
        protocol_digest=STAGE_A_DIGEST,
        geometry_digest="0" * 64,
        code_digest="1" * 64,
    )

    assert result["experiment_id"] == "EXP-290"
    assert result["evidence_level"] == "EV-E2"
    assert result["scientific_evidence_eligible"] is False
    assert result["confirmatory_data_consumed"] is False
    assert result["challenge_materialized"] is False
    assert result["promotion_claimed"] is False
    assert result["evaluation_mapping_used_for_training"] is False
    assert result["evaluation_mapping_used_by_learned_mode"] is False
    assert result["oracle_transfer_mode_deployable"] is False

    modes = result["evaluation"]["modes"]
    assert set(modes) == {
        "LOCAL_ONLY_CONTROL",
        "LEARNED_STRUCTURAL_TRANSFER",
        "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND",
    }
    digests = set()
    slot_capacities = set()
    for name, receipt in modes.items():
        assert receipt["target_local_memory_enabled"] is True
        assert receipt["transfer_scorer_executed"] is True
        assert receipt["fixed_transferred_slot_accounting"] is True
        assert receipt["target_local_insertion_requires_observed_contradiction"] is True
        assert receipt["evaluator_truth_gates_target_action"] is False
        digests.add(receipt["model_digest"])
        slot_capacities.add(receipt["transferred_slot_capacity"])
        if name == "LEARNED_STRUCTURAL_TRANSFER":
            assert receipt["oracle_mapping_delivered"] is False
            assert receipt["evaluation_mapping_delivered"] is False
        if name == "ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND":
            assert receipt["oracle_mapping_delivered"] is True
    assert len(digests) == 1
    assert len(slot_capacities) == 1
    assert result["post_training_model_digest"] == result["post_evaluation_model_digest"]


def test_exp290_tiny_root_preserves_nonidentity_surface_and_first_encounter_endpoint() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import run_exp290_root

    result = run_exp290_root(
        canonical_index=1,
        geometry=_tiny_geometry(),
        protocol_digest=STAGE_A_DIGEST,
        geometry_digest="2" * 64,
        code_digest="3" * 64,
    )
    metrics = result["root_metrics"]
    assert metrics["heldout_nonidentity_permutation_rate"] == pytest.approx(1.0)
    assert metrics["primary_endpoint"] == "source_equivalent_target_dead_end_rate"
    assert metrics["epsilon_denominator_rescue_used"] is False
    assert metrics["local_only_transfer_activated"] is False
    assert metrics["oracle_transfer_mode_deployable"] is False
    assert metrics["learned_transferred_prune_count"] >= 0
    assert 0.0 <= metrics["learned_valid_state_overprune_rate"] <= 1.0
    assert 0.0 <= metrics["learned_verified_solution_rate"] <= 1.0
    assert 0.0 <= metrics["control_verified_solution_rate"] <= 1.0


def test_exp290_root_decision_is_recomputed_from_frozen_primitives() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import (
        classify_exp290_root,
        run_exp290_root,
    )

    result = run_exp290_root(
        canonical_index=2,
        geometry=_tiny_geometry(),
        protocol_digest=STAGE_A_DIGEST,
        geometry_digest="4" * 64,
        code_digest="5" * 64,
    )
    assert result["decision"] == classify_exp290_root(result["root_metrics"])


def test_exp290_classifier_requires_every_frozen_success_predicate() -> None:
    from nolane_ai.experiments.exp290_structural_clause_transfer import classify_exp290_root

    favorable = {
        "oracle_headroom": 0.5,
        "learned_headroom": 0.3,
        "learned_oracle_value_capture": 0.6,
        "learned_valid_state_overprune_rate": 0.0,
        "control_verified_solution_rate": 1.0,
        "learned_verified_solution_rate": 1.0,
        "learned_transferred_prune_count": 2,
        "heldout_nonidentity_permutation_rate": 1.0,
        "evaluation_mapping_used_by_learned_mode": False,
        "oracle_mapping_used_by_learned_mode": False,
    }
    assert classify_exp290_root(favorable) == "LEARNED_STRUCTURAL_TRANSFER_ESTABLISHED"

    oracle_missing = dict(favorable, oracle_headroom=0.0)
    assert classify_exp290_root(oracle_missing) == "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"

    for key, value in (
        ("learned_headroom", 0.0),
        ("learned_oracle_value_capture", 0.49),
        ("learned_valid_state_overprune_rate", 0.006),
        ("learned_verified_solution_rate", 0.989),
        ("learned_transferred_prune_count", 0),
        ("heldout_nonidentity_permutation_rate", 0.99),
        ("evaluation_mapping_used_by_learned_mode", True),
        ("oracle_mapping_used_by_learned_mode", True),
    ):
        candidate = dict(favorable)
        candidate[key] = value
        assert classify_exp290_root(candidate) == "LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED"


def test_exp290_learned_control_source_has_no_evaluator_mapping_access() -> None:
    from nolane_ai.experiments import exp290_structural_clause_transfer as runner

    learned_source = inspect.getsource(runner._translate_learned_source_clauses)
    assert "evaluator_only_source_to_target_variable_permutation" not in learned_source
    assert "oracle" not in learned_source.lower()
    assert "solution_targets" not in learned_source
