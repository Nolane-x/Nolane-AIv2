from __future__ import annotations

from itertools import product
from typing import Any


NULL_TRANSFER_CONTROL = "NULL_TRANSFER_CONTROL"
RAW_SURFACE_TRANSFER_CONTROL = "RAW_SURFACE_TRANSFER_CONTROL"
LEARNED_STRUCTURAL_TRANSFER = "LEARNED_STRUCTURAL_TRANSFER"
ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND = "ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND"

_EXP290_MODES = {
    NULL_TRANSFER_CONTROL,
    RAW_SURFACE_TRANSFER_CONTROL,
    LEARNED_STRUCTURAL_TRANSFER,
    ORACLE_STRUCTURAL_TRANSFER_UPPER_BOUND,
}


def _canonical_clause(clause: list[list[Any]]) -> list[list[Any]]:
    return [
        [str(name), int(value)]
        for name, value in sorted(
            ((str(name), int(value)) for name, value in clause),
            key=lambda item: item[0],
        )
    ]


def _clause_key(clause: list[list[Any]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(name), int(value)) for name, value in _canonical_clause(clause))


def _public_contradiction(
    problem: dict[str, Any],
    assignment: dict[str, int],
) -> bool:
    for constraint in problem["constraints"]:
        scope = [str(name) for name in constraint["scope"]]
        if not all(name in assignment for name in scope):
            continue
        values = tuple(int(assignment[name]) for name in scope)
        allowed = {
            tuple(int(value) for value in row)
            for row in constraint["allowed"]
        }
        if values not in allowed:
            return True
    return False


def _assignment_is_complete_and_valid(
    problem: dict[str, Any],
    assignment: dict[str, int],
) -> bool:
    names = [str(variable["name"]) for variable in problem["variables"]]
    return set(assignment) == set(names) and not _public_contradiction(problem, assignment)


def _has_valid_completion(
    problem: dict[str, Any],
    partial_assignment: dict[str, int],
) -> bool:
    variables = [str(variable["name"]) for variable in problem["variables"]]
    remaining = [name for name in variables if name not in partial_assignment]
    if _public_contradiction(problem, partial_assignment):
        return False
    for values in product((0, 1), repeat=len(remaining)):
        candidate = dict(partial_assignment)
        candidate.update(dict(zip(remaining, values, strict=True)))
        if _assignment_is_complete_and_valid(problem, candidate):
            return True
    return False


def _matching_clause(
    clauses: list[list[list[Any]]],
    assignment: dict[str, int],
) -> list[list[Any]] | None:
    for clause in clauses:
        if all(assignment.get(str(name)) == int(value) for name, value in clause):
            return clause
    return None


def map_source_clause_with_learned_indices(
    source_clause: list[list[Any]],
    *,
    source_names: list[str],
    target_names: list[str],
    learned_source_to_target: list[int],
) -> tuple[list[list[Any]] | None, str]:
    if len(learned_source_to_target) != len(source_names):
        raise ValueError("learned mapping length must equal source variable count")
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise ValueError("source and target surface names must be unique")
    source_index = {str(name): index for index, name in enumerate(source_names)}
    mapped: list[list[Any]] = []
    used_target_indices: set[int] = set()
    for name, value in source_clause:
        key = str(name)
        if key not in source_index:
            raise ValueError("source clause references unknown source variable")
        target_index = int(learned_source_to_target[source_index[key]])
        if target_index < 0 or target_index >= len(target_names):
            raise ValueError("learned mapping target index out of range")
        if target_index in used_target_indices:
            return None, "mapping_collision"
        used_target_indices.add(target_index)
        mapped.append([str(target_names[target_index]), int(value)])
    return _canonical_clause(mapped), "accepted"


def _build_learned_transferred_clauses(
    *,
    source_clauses: list[list[list[Any]]],
    source_names: list[str],
    target_names: list[str],
    learned_source_to_target: list[int],
) -> tuple[list[list[list[Any]]], list[dict[str, Any]]]:
    accepted: list[list[list[Any]]] = []
    receipts: list[dict[str, Any]] = []
    for source_clause in source_clauses:
        mapped, reason = map_source_clause_with_learned_indices(
            source_clause,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=learned_source_to_target,
        )
        receipts.append(
            {
                "source_clause": _canonical_clause(source_clause),
                "mapped_target_clause": mapped,
                "status": reason,
                "oracle_fallback_used": False,
                "evaluator_truth_used": False,
            }
        )
        if mapped is not None:
            accepted.append(mapped)
    return accepted, receipts


def _build_oracle_transferred_clauses(
    *,
    pair: dict[str, Any],
    source_clauses: list[list[list[Any]]],
) -> list[list[list[Any]]]:
    source_names = [str(name) for name in pair["source_surface_names"]]
    target_names = [str(name) for name in pair["target_surface_names"]]
    mapping = [int(index) for index in pair["hidden_source_to_target_index"]]
    transferred: list[list[list[Any]]] = []
    for source_clause in source_clauses:
        mapped, reason = map_source_clause_with_learned_indices(
            source_clause,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=mapping,
        )
        if mapped is None or reason != "accepted":
            raise RuntimeError("oracle source-to-target bijection produced an invalid clause")
        transferred.append(mapped)
    return transferred


def acquire_source_clauses(
    *,
    pair: dict[str, Any],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")
    problem = pair["source_problem"]
    source_names = [str(name) for name in pair["source_surface_names"]]
    clauses: list[list[list[Any]]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    receipts: list[dict[str, Any]] = []
    steps = 0

    for restart_index in range(int(restart_orders.shape[0])):
        assignment: dict[str, int] = {}
        restart_terminated = False
        for variable_tensor in restart_orders[restart_index]:
            variable_index = int(variable_tensor.item())
            variable_name = source_names[variable_index]
            accepted = False
            for value_tensor in restart_value_orders[restart_index, variable_index]:
                if steps >= max_search_steps:
                    restart_terminated = True
                    break
                steps += 1
                value = int(value_tensor.item())
                candidate = dict(assignment)
                candidate[variable_name] = value
                if _public_contradiction(problem, candidate):
                    if len(candidate) == 2:
                        clause = _canonical_clause(
                            [[name, literal_value] for name, literal_value in candidate.items()]
                        )
                        key = _clause_key(clause)
                        inserted = key not in seen
                        if inserted:
                            seen.add(key)
                            clauses.append(clause)
                        receipts.append(
                            {
                                "restart_index": restart_index,
                                "source_clause": clause,
                                "partial_assignment_reached": True,
                                "dead_end_observed_before_insertion": True,
                                "inserted": inserted,
                                "evaluator_truth_used_for_insertion": False,
                                "oracle_conflict_core_used": False,
                            }
                        )
                    restart_terminated = True
                    break
                assignment = candidate
                accepted = True
                break
            if restart_terminated or steps >= max_search_steps or not accepted:
                break

    return {
        "clauses": clauses,
        "receipts": receipts,
        "source_search_steps": int(steps),
        "source_clause_count": len(clauses),
        "evaluator_truth_used_for_insertion": False,
    }


def _run_target_search(
    *,
    problem: dict[str, Any],
    target_names: list[str],
    transferred_clauses: list[list[list[Any]]],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    steps = 0
    clause_comparisons = 0
    candidate_evaluation_operations = 0
    transferred_clause_hits = 0
    observed_dead_ends: list[list[list[Any]]] = []
    prune_receipts: list[dict[str, Any]] = []
    completed_valid_restarts = 0
    candidate_step_cost = len(problem["variables"]) + len(problem["constraints"])

    for restart_index in range(int(restart_orders.shape[0])):
        assignment: dict[str, int] = {}
        restart_terminated = False
        for variable_tensor in restart_orders[restart_index]:
            variable_index = int(variable_tensor.item())
            variable_name = target_names[variable_index]
            accepted = False
            for value_tensor in restart_value_orders[restart_index, variable_index]:
                if steps >= max_search_steps:
                    restart_terminated = True
                    break
                steps += 1
                candidate_evaluation_operations += candidate_step_cost
                value = int(value_tensor.item())
                candidate = dict(assignment)
                candidate[variable_name] = value

                clause_comparisons += len(transferred_clauses)
                matched = _matching_clause(transferred_clauses, candidate)
                if matched is not None:
                    transferred_clause_hits += 1
                    prune_receipts.append(
                        {
                            "restart_index": restart_index,
                            "candidate_assignment": _canonical_clause(
                                [[name, literal] for name, literal in candidate.items()]
                            ),
                            "matched_clause": _canonical_clause(matched),
                        }
                    )
                    continue

                if _public_contradiction(problem, candidate):
                    if len(candidate) == 2:
                        observed_dead_ends.append(
                            _canonical_clause(
                                [[name, literal] for name, literal in candidate.items()]
                            )
                        )
                    restart_terminated = True
                    break

                assignment = candidate
                accepted = True
                break
            if restart_terminated or steps >= max_search_steps or not accepted:
                break

        if _assignment_is_complete_and_valid(problem, assignment):
            completed_valid_restarts += 1
            break

    return {
        "search_steps": int(steps),
        "candidate_step_cost": int(candidate_step_cost),
        "candidate_evaluation_operations": int(candidate_evaluation_operations),
        "clause_comparisons": int(clause_comparisons),
        "transferred_clause_hit_count": int(transferred_clause_hits),
        "observed_dead_ends": observed_dead_ends,
        "prune_receipts": prune_receipts,
        "completed_valid_restarts": int(completed_valid_restarts),
        "verified_solution": completed_valid_restarts > 0,
    }


def run_target_mode(
    *,
    mode: str,
    pair: dict[str, Any],
    source_clauses: list[list[list[Any]]],
    learned_source_to_target: list[int],
    restart_orders: Any,
    restart_value_orders: Any,
    max_search_steps: int,
) -> dict[str, Any]:
    if mode not in _EXP290_MODES:
        raise ValueError(f"unknown EXP-290 transfer mode: {mode}")
    if max_search_steps <= 0:
        raise ValueError("max_search_steps must be positive")

    source_names = [str(name) for name in pair["source_surface_names"]]
    target_names = [str(name) for name in pair["target_surface_names"]]
    if set(source_names) & set(target_names):
        raise ValueError("EXP-290 source and target surface namespaces must be disjoint")

    mapping_receipts: list[dict[str, Any]] = []
    oracle_correspondence_delivered = False
    if mode == NULL_TRANSFER_CONTROL:
        transferred: list[list[list[Any]]] = []
    elif mode == RAW_SURFACE_TRANSFER_CONTROL:
        transferred = [_canonical_clause(clause) for clause in source_clauses]
    elif mode == LEARNED_STRUCTURAL_TRANSFER:
        transferred, mapping_receipts = _build_learned_transferred_clauses(
            source_clauses=source_clauses,
            source_names=source_names,
            target_names=target_names,
            learned_source_to_target=learned_source_to_target,
        )
    else:
        transferred = _build_oracle_transferred_clauses(
            pair=pair,
            source_clauses=source_clauses,
        )
        oracle_correspondence_delivered = True

    search = _run_target_search(
        problem=pair["target_problem"],
        target_names=target_names,
        transferred_clauses=transferred,
        restart_orders=restart_orders,
        restart_value_orders=restart_value_orders,
        max_search_steps=max_search_steps,
    )

    manifest_keys = {
        _clause_key(item["mapped_target_clause"])
        for item in pair["target_transfer_opportunities"]
    }
    structural_reentries = sum(
        1
        for clause in search["observed_dead_ends"]
        if _clause_key(clause) in manifest_keys
    )
    invalid_prunes = 0
    for prune in search["prune_receipts"]:
        partial = {
            str(name): int(value)
            for name, value in prune["candidate_assignment"]
        }
        if _has_valid_completion(pair["target_problem"], partial):
            invalid_prunes += 1
    prune_count = len(search["prune_receipts"])
    overprune_rate = float(invalid_prunes / prune_count) if prune_count else 0.0

    return {
        "mode": mode,
        "transferred_clause_count": len(transferred),
        "mapping_receipts": mapping_receipts,
        "mapping_collision_count": sum(
            1 for row in mapping_receipts if row["status"] == "mapping_collision"
        ),
        "transferred_clause_hit_count": int(search["transferred_clause_hit_count"]),
        "structural_repeat_dead_end_reentries": int(structural_reentries),
        "predeclared_transfer_opportunities": len(manifest_keys),
        "structural_repeat_dead_end_rate": (
            float(structural_reentries / len(manifest_keys)) if manifest_keys else None
        ),
        "invalid_transferred_prune_count": int(invalid_prunes),
        "transferred_prune_event_count": int(prune_count),
        "valid_state_overprune_rate": overprune_rate,
        "verified_solution": bool(search["verified_solution"]),
        "verified_solution_rate": 1.0 if search["verified_solution"] else 0.0,
        "search_steps": int(search["search_steps"]),
        "candidate_step_cost": int(search["candidate_step_cost"]),
        "candidate_evaluation_operations": int(search["candidate_evaluation_operations"]),
        "clause_comparisons": int(search["clause_comparisons"]),
        "search_accounted_operations": int(
            search["candidate_evaluation_operations"] + search["clause_comparisons"]
        ),
        "target_local_clause_learning_enabled": False,
        "oracle_correspondence_delivered": oracle_correspondence_delivered,
        "evaluator_validity_truth_delivered": False,
        "target_clause_truth_delivered": False,
        "surface_namespaces_disjoint": True,
        "raw_surface_transfer_hit_count": (
            int(search["transferred_clause_hit_count"])
            if mode == RAW_SURFACE_TRANSFER_CONTROL
            else 0
        ),
        "posthoc_safety_audit_only": True,
    }


def classify_exp290_root(metrics: dict[str, object]) -> str:
    oracle_headroom = float(metrics["oracle_headroom"])
    oracle_repeat_reduction = float(metrics["oracle_structural_repeat_relative_reduction"])
    if oracle_headroom <= 0.0 or oracle_repeat_reduction < 0.25:
        return "ORACLE_TRANSFER_HEADROOM_NOT_REPLICATED"

    null_repeat = float(metrics["null_structural_repeat_dead_end_rate"])
    learned_repeat = float(metrics["learned_structural_repeat_dead_end_rate"])
    null_solution = float(metrics["null_verified_solution_rate"])
    learned_solution = float(metrics["learned_verified_solution_rate"])
    learned_ok = all(
        (
            float(metrics["learned_headroom"]) > 0.0,
            float(metrics["learned_oracle_value_capture"]) >= 0.50,
            float(metrics["learned_source_to_target_correspondence_accuracy"]) > 0.50,
            float(metrics["learned_exact_transferred_clause_recovery_rate"]) > 0.50,
            learned_repeat < null_repeat,
            learned_solution >= null_solution - 0.01,
            float(metrics["learned_valid_state_overprune_rate"]) <= 0.005,
            not bool(metrics["learned_oracle_correspondence_delivered"]),
            not bool(metrics["learned_evaluator_validity_truth_delivered"]),
            not bool(metrics["learned_target_clause_truth_delivered"]),
            int(metrics["raw_surface_transfer_hit_count"]) == 0,
            bool(metrics["surface_namespaces_disjoint"]),
        )
    )
    return (
        "LEARNED_CLAUSE_TRANSFER_ESTABLISHED"
        if learned_ok
        else "LEARNED_CLAUSE_TRANSFER_NOT_ESTABLISHED"
    )
