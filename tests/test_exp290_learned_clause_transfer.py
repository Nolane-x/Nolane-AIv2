from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")


def _favorable_metrics() -> dict[str, object]:
    return {
        "null_cost": 100.0,
        "raw_cost": 100.0,
        "learned_cost": 80.0,
        "oracle_cost": 70.0,
        "oracle_headroom": 30.0,
        "learned_headroom": 20.0,
        "learned_oracle_value_capture": 2.0 / 3.0,
        "null_structural_repeat_dead_end_rate": 1.0,
        "oracle_structural_repeat_dead_end_rate": 0.5,
        "oracle_structural_repeat_relative_reduction": 0.5,
        "learned_structural_repeat_dead_end_rate": 0.5,
        "learned_source_to_target_correspondence_accuracy": 0.75,
        "learned_exact_transferred_clause_recovery_rate": 0.75,
        "null_verified_solution_rate": 1.0,
        "learned_verified_solution_rate": 1.0,
        "learned_valid_state_overprune_rate": 0.0,
        "learned_oracle_correspondence_delivered": False,
        "learned_evaluator_validity_truth_delivered": False,
        "learned_target_clause_truth_delivered": False,
        "raw_surface_transfer_hit_count": 0,
        "surface_namespaces_disjoint": True,
    }


def test_exp290_root_classifier_requires_all_frozen_predicates() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import classify_exp290_root

    metrics = _favorable_metrics()
    assert classify_exp290_root(metrics) == "LEARNED_CLAUSE_TRANSFER_ESTABLISHED"

    for field, value in (
        ("learned_headroom", 0.0),
        ("learned_oracle_value_capture", 0.49),
        ("learned_source_to_target_correspondence_accuracy", 0.50),
        ("learned_exact_transferred_clause_recovery_rate", 0.50),
        ("learned_structural_repeat_dead_end_rate", 1.0),
        ("learned_verified_solution_rate", 0.989),
        ("learned_valid_state_overprune_rate", 0.006),
        ("raw_surface_transfer_hit_count", 1),
        ("surface_namespaces_disjoint", False),
    ):
        candidate = dict(metrics)
        candidate[field] = value
        assert classify_exp290_root(candidate) == "LEARNED_CLAUSE_TRANSFER_NOT_ESTABLISHED"


def test_exp290_root_classifier_distinguishes_missing_oracle_transfer_headroom() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import classify_exp290_root

    metrics = _favorable_metrics()
    metrics["oracle_headroom"] = 0.0
    assert classify_exp290_root(metrics) == "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"

    metrics = _favorable_metrics()
    metrics["oracle_structural_repeat_relative_reduction"] = 0.24
    assert classify_exp290_root(metrics) == "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"


def test_exp290_learned_clause_mapping_rejects_collisions_without_fallback() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import map_source_clause_with_learned_indices

    source_names = ["s0", "s1", "s2"]
    target_names = ["t0", "t1", "t2"]
    clause = [["s0", 1], ["s1", 0]]

    mapped, reason = map_source_clause_with_learned_indices(
        clause,
        source_names=source_names,
        target_names=target_names,
        learned_source_to_target=[2, 1, 0],
    )
    assert mapped == [["t2", 1], ["t1", 0]]
    assert reason == "accepted"

    mapped, reason = map_source_clause_with_learned_indices(
        clause,
        source_names=source_names,
        target_names=target_names,
        learned_source_to_target=[2, 2, 0],
    )
    assert mapped is None
    assert reason == "mapping_collision"


def test_exp290_source_clause_acquisition_requires_observed_two_literal_dead_end() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import acquire_source_clauses
    from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator

    batch = Exp290TransferGenerator(root_seed="20260913-exp290-runner-test").make_batch(
        replicate=10000,
        rng_stream="evaluation",
        device="cpu",
    )
    pair = batch.metadata["pairs"][0]
    result = acquire_source_clauses(
        pair=pair,
        restart_orders=batch.source_restart_orders[0],
        restart_value_orders=batch.source_restart_value_orders[0],
        max_search_steps=24,
    )

    assert result["clauses"]
    assert all(len(clause) == 2 for clause in result["clauses"])
    assert all(row["partial_assignment_reached"] is True for row in result["receipts"])
    assert all(row["dead_end_observed_before_insertion"] is True for row in result["receipts"])
    assert all(row["evaluator_truth_used_for_insertion"] is False for row in result["receipts"])


def test_exp290_oracle_and_correct_learned_transfer_prevent_structural_reentries_and_reduce_search_work() -> None:
    from nolane_ai.experiments.exp290_learned_clause_transfer import (
        LEARNED_STRUCTURAL_TRANSFER,
        NULL_TRANSFER_CONTROL,
        ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND,
        RAW_SURFACE_TRANSFER_CONTROL,
        acquire_source_clauses,
        run_target_mode,
    )
    from nolane_ai.experiments.exp290_transfer_worlds import Exp290TransferGenerator

    batch = Exp290TransferGenerator(root_seed="20260913-exp290-target-test").make_batch(
        replicate=10000,
        rng_stream="evaluation",
        device="cpu",
    )
    pair = batch.metadata["pairs"][0]
    source = acquire_source_clauses(
        pair=pair,
        restart_orders=batch.source_restart_orders[0],
        restart_value_orders=batch.source_restart_value_orders[0],
        max_search_steps=24,
    )
    correct_mapping = pair["hidden_source_to_target_index"]

    kwargs = {
        "pair": pair,
        "source_clauses": source["clauses"],
        "learned_source_to_target": correct_mapping,
        "restart_orders": batch.target_restart_orders[0],
        "restart_value_orders": batch.target_restart_value_orders[0],
        "max_search_steps": 24,
    }
    null = run_target_mode(mode=NULL_TRANSFER_CONTROL, **kwargs)
    raw = run_target_mode(mode=RAW_SURFACE_TRANSFER_CONTROL, **kwargs)
    learned = run_target_mode(mode=LEARNED_STRUCTURAL_TRANSFER, **kwargs)
    oracle = run_target_mode(mode=ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND, **kwargs)

    assert null["structural_repeat_dead_end_reentries"] > 0
    assert raw["transferred_clause_hit_count"] == 0
    assert raw["structural_repeat_dead_end_reentries"] == null["structural_repeat_dead_end_reentries"]
    assert learned["structural_repeat_dead_end_reentries"] < null["structural_repeat_dead_end_reentries"]
    assert oracle["structural_repeat_dead_end_reentries"] < null["structural_repeat_dead_end_reentries"]
    assert learned["search_accounted_operations"] < null["search_accounted_operations"]
    assert oracle["search_accounted_operations"] < null["search_accounted_operations"]
    assert learned["target_local_clause_learning_enabled"] is False
    assert learned["oracle_correspondence_delivered"] is False
    assert oracle["oracle_correspondence_delivered"] is True


def test_exp290_learned_target_control_does_not_read_evaluator_mapping_or_solution_truth() -> None:
    from nolane_ai.experiments import exp290_learned_clause_transfer as runner

    learned_source = inspect.getsource(runner._build_learned_transferred_clauses)
    target_source = inspect.getsource(runner._run_target_search)
    assert "hidden_source_to_target_index" not in learned_source
    assert "evaluator_solution" not in learned_source
    assert "hidden_source_to_target_index" not in target_source
    assert "evaluator_solution" not in target_source
